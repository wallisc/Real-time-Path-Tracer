#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <float.h>

#include "glm/gtc/matrix_transform.hpp"

#include "Camera.h"
#include "Sphere.h"
#include "Triangle.h"
#include "glm/glm.hpp"
#include "PhongShader.h"
#include "cudaError.h"
#include "kernel.h"

#include "bvh.cpp"

#define kNoShapeFound NULL

const float kMaxDist = FLT_MAX;
using glm::vec3;
using glm::vec4;
using glm::mat4;
using std::vector;
using std::pair;

texture<uchar4, 2, cudaReadModeNormalizedFloat> mytex;

// Only works with 24 bit images that are a power of 2
unsigned char* readBMP(char* filename, int *retWidth, int *retHeight)
{
   int i;
   FILE* f = fopen(filename, "rb");
   unsigned char info[54];
   fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

   // extract image height and width from header
   int width = *(int*)&info[18];
   int height = *(int*)&info[22];

   int size = 3 * width * height;
   unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
   unsigned char* retData = new unsigned char[size + width * height]; // allocate 4 bytes per pixel
   fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
   fclose(f);

   for(i = 0; i < width * height; i++)
   {
      retData[4 * i] = data[3 * i + 2];
      retData[4 * i + 1] = data[3 * i + 1];
      retData[4 * i + 2] = data[3 * i];
      retData[4 * i + 3] = 0;
   }

   delete data;
   *retWidth = width;
   *retHeight = height;
   return retData;
}

const bool kLeft = 0;
const bool kRight = 1;

typedef struct StackEntry {
   bool nextDir;
   BVHNode *node;
   __device__ StackEntry(BVHNode *stackNode = NULL, char nextDirection = 0) : node(stackNode), nextDir(nextDirection) {}
} StackEntry;

// Find the closest shape. The index of the intersecting object is stored in
// retOjIdx and the t-value along the input ray is stored in retParam
//
// toBeat can be set to a float value if you want to short-circuit as soon
// as you find an object closer than toBeat
//
// If no intersection is found, retObjIdx is set to 'kNoShapeFound'
__device__ void getClosestIntersection(const Ray &ray, BVHTree *tree, 
      Triangle **retObj, float *retParam, float toBeat = -FLT_MAX) {
   float t = kMaxDist;
   Triangle *closestGeom = kNoShapeFound;
   int maxDepth = 0;

   StackEntry stack[kMaxStackSize];
   int stackSize = 0;
   bool justPoppedStack = false;

   BVHNode *cursor = tree->root;
   bool nextDir;
     
   do {
      if (stackSize >= kMaxStackSize) {
         printf("Stack full, aborting!\n");
         return;
      }
         
      // If at a leaf
      if (cursor->geom) {
         maxDepth = max(maxDepth, stackSize);
         float dist = cursor->geom->getIntersection(ray);
         //If two shapes are overlapping, pick the one with the closest facing normal
         if (isFloatEqual(t, dist)) {
            glm::vec3 oldNorm = closestGeom->getNormalAt(ray, t);
            glm::vec3 newNorm = cursor->geom->getNormalAt(ray, dist);
            glm::vec3 eye = glm::normalize(-ray.d);
            float newDot = glm::dot(eye, newNorm);
            float oldDot = glm::dot(eye, oldNorm);
            if (newDot > oldDot) {
               closestGeom = cursor->geom;
               t = dist;
               if (t < toBeat) {
                  *retObj = closestGeom;
                  *retParam = t;
                  return;
               }
            }
         // Otherwise, if one face is front of the current one
         } else {
            if (dist < t && isFloatAboveZero(dist)) {
               t = dist;
               closestGeom = cursor->geom;
               if (t < toBeat) {
                  *retObj = closestGeom;
                  *retParam = t;
                  return;
               }
            }
         }
      // If not on a leaf and neither branch has been explored
      } else if (!justPoppedStack) { 
         float left = cursor->left->bb.getIntersection(ray);

         if (!cursor->right && isFloatAboveZero(left) && left < t) {
            cursor = cursor->left;
            justPoppedStack = false;
            continue;
         }

         // Go down the tree with the closest bounding box
         float right = cursor->right->bb.getIntersection(ray);

         if (isFloatAboveZero(right) && (right <= left || !isFloatAboveZero(left)) && right < t) {
            if (isFloatAboveZero(left)) stack[stackSize++] = StackEntry(cursor, kLeft);
            cursor = cursor->right;
            justPoppedStack = false;
            continue;
         } else if (isFloatAboveZero(left) && (left <= right || !isFloatAboveZero(right)) && left < t) {
            if (isFloatAboveZero(right)) stack[stackSize++] = StackEntry(cursor, kRight);
            cursor = cursor->left;
            justPoppedStack = false;
            continue;
         } 
      // If coming back from a 'recursion' and one of the branches hasn't been explored
      } else {
         if (nextDir == kRight) {
            float right = cursor->right->bb.getIntersection(ray);
            if (right < t) {
               cursor = cursor->right;
               justPoppedStack = false;
               continue;
            }
         } else {
            float left = cursor->left->bb.getIntersection(ray);
            if (left < t) {
               cursor = cursor->left;
               justPoppedStack = false;
               continue;
            }
         }
      }

      if(stackSize == 0) {
         break;
      }

      // Pop the stack
      cursor = stack[stackSize - 1].node; 
      nextDir = stack[stackSize - 1].nextDir;
      justPoppedStack = true;
      stackSize--;
   } while(true);

   *retObj = closestGeom;
   *retParam = t;
}

__device__ bool isInShadow(const Ray &shadow, BVHTree *tree, float intersectParam) {
   float closestIntersect;
   Triangle *closestObj;
   getClosestIntersection(shadow, tree, &closestObj, &closestIntersect, intersectParam);
   return isFloatLessThan(closestIntersect, intersectParam);
}

__device__ vec3 cosineWeightedSample(vec3 normal, float rand1, float rand2) {
   float distFromCenter = rand1;
   float theta = 2.0f * M_PI * rand2;
   float phi = M_PI / 2.0f - acos(distFromCenter);

   float phiDeg = phi * 180.0f / M_PI;
   float thetaDeg = theta * 180.0f / M_PI;

   vec3 outV; 
   if (normal.x > .99f) outV = vec3(0.0f, 1.0f, 0.0f);
   else if (normal.x < -.99f) outV = vec3(0.0f, -1.0f, 0.0f);
   else outV = glm::cross(normal, vec3(1.0f, 0.0, 0.0));
   glm::mat4 rot1 = glm::rotate(glm::mat4(1.0f), phiDeg, outV);
   glm::mat4 rot2 = glm::rotate(glm::mat4(1.0f), thetaDeg, normal);
   glm::vec4 norm(normal.x, normal.y, normal.z, 0.0f);
   
   return vec3(rot2 * rot1 * norm);
}


__device__ glm::vec3 getColor(Triangle *geom, Ray ray, float param) {
   Material m = geom->getMaterial();
   if (m.texId == NO_TEXTURE) {
      return m.clr;
   } else {
      glm::vec2 uv = geom->UVAt(ray, param);
      float4 clr = tex2D(mytex, uv.x, uv.y);
      return vec3(clr.x, clr.y, clr.z);
   }
}

//Note: The ray parameter must stay as a copy (not a reference) 
__device__ vec3 shadeObject(BVHTree *tree, 
      Triangle *lights[], int lightCount, Triangle* geom, 
      float intParam, Ray ray, curandState *randStates) {

   Material m = geom->getMaterial();
   if (m.emissive) { return m.clr;}
   if (isFloatZero(1.0f - m.refl - m.alpha)) return glm::vec3(0.0f); 


   glm::vec3 intersectPoint = ray.getPoint(intParam);
   vec3 normal = geom->getNormalAt(ray, intParam);
   vec3 matClr = getColor(geom, ray, intParam);
   vec3 eyeVec = glm::normalize(-ray.d);
   vec3 totalLight(0.0f);

   for(int lightIdx = 0; lightIdx < lightCount; lightIdx++) {

      vec3 lightColor = lights[lightIdx]->mat.clr;
      //if (threadIdx.x == 0 && blockIdx.x == 0) printf("%f, %f, %f\n", lightColor.x, lightColor.y, lightColor.z);

      // Randomly sample the area light with a random baryocentric coordinate
      float alpha = curand_uniform(randStates);
      float beta = curand_uniform(randStates) * (1.0f - alpha);
      vec3 lightPos = lights[lightIdx]->getPointFromBary(alpha, beta);

      vec3 lightDir = glm::normalize(lightPos - intersectPoint);
      Ray shadow(lightPos, -lightDir);
      shadow.o += shadow.d * BIG_EPSILON;
      float intersectParam = geom->getIntersection(shadow);
      bool inShadow = isInShadow(shadow, tree, intersectParam); 

      totalLight += PhongShader::shade(matClr, m.amb, m.dif, m.spec, m.rough, 
            eyeVec, lightDir, lightColor, normal, inShadow);
   }

   return totalLight * (1.0f - m.refl - m.alpha);
}

__global__ void initScene(Triangle geomList[], TKTriangle *triangleTks, 
      int numTris, TKSmoothTriangle *smthTriTks, int numSmthTris, ShadingType stype) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int gridSize = gridDim.x * blockDim.x;
   int geomListSize = 0;

   for (int triIdx = idx; triIdx < numTris; triIdx += gridSize) {
      const TKTriangle &t = triangleTks[triIdx];
      if (t.mod.fin.em) {
         Material m(t.mod.pig.clr);
         geomList[triIdx + geomListSize] = Triangle(t.p1, t.p2, t.p3, t.n, t.n, t.n, m, t.vt1, t.vt2, t.vt3);
      } else {  
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior, t.mod.pig.texId);
         geomList[triIdx + geomListSize] = Triangle(t.p1, t.p2, t.p3, t.n, t.n, t.n, m, t.vt1, t.vt2, t.vt3);
      }

   }
   geomListSize += numTris;

   for (int smTriIdx = idx; smTriIdx < numSmthTris; smTriIdx += gridSize) {
      const TKSmoothTriangle &t = smthTriTks[smTriIdx];
      if (t.mod.fin.em) {
         Material m(t.mod.pig.clr);
         geomList[smTriIdx + geomListSize] = Triangle(t.p1, t.p2, t.p3, t.n1, t.n2, t.n3, m, t.vt1, t.vt2, t.vt3);
      } else {
         const TKFinish f = t.mod.fin;
         Material m(t.mod.pig.clr, t.mod.pig.f, f.amb, f.dif, f.spec, f.rough, f.refl, f.refr, f.ior, t.mod.pig.texId);
         geomList[smTriIdx + geomListSize] = Triangle(t.p1, t.p2, t.p3, t.n1, t.n2, t.n3, 
               m, t.vt1, t.vt2, t.vt3);
      }

   }

   geomListSize += numSmthTris;

}
__global__ void setupLights(Triangle geomList[], int listSize, Triangle *lights[]) {
   int lightNum = 0;
   for (int i = 0; i < listSize; i++) {
      if (geomList[i].getMaterial().emissive) {
         lights[lightNum++] = geomList + i;
      }
   }
}

__global__ void initCurand(curandState randStates[], int numRandStates) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if (x >= numRandStates) return;

   curand_init(x, 0, 0, &randStates[x]);
}

__global__ void generateCameraRays(int resWidth, int resHeight, Camera cam, RayCache cache[], curandState randStates[]) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   float celWidth = 1.0f / (float)resWidth;
   float celHeight = 1.0f / (float)resHeight;

   if (x >= resWidth || y >= resWidth) return;

   int tid = threadIdx.y * blockDim.x + threadIdx.x;

   float uJitter = (curand_uniform(&randStates[tid]) - .5f) * celWidth;
   float vJitter = (curand_uniform(&randStates[tid]) - .5f) * celHeight;
   float u = ((float)(x - resWidth / 2) + .5f) * celWidth * 2.0f + uJitter; 
   float v = ((float)(y - resHeight / 2) + .5f) * celHeight * 2.0f + vJitter; 

   // .5f is because the magnitude of cam.right and cam.up should be equal
   // to the width and height of the image plane in world space
   vec3 rPos = u *.5f * cam.right + v * .5f * cam.up + cam.pos;
   vec3 rDir = rPos - cam.pos + cam.lookAtDir;
   int index = y * resWidth + x;
   cache[index] = RayCache(Ray(rPos, rDir), glm::vec3(1.0f));
}

__global__ void rayTrace(int column, int row, int resWidth, int resHeight,
      BVHTree *tree, Triangle *lights[], int lightCount,  
      vec3 output[], curandState randStates[], RayCache cache[], int depth) {

   int tid = threadIdx.y * blockDim.x + threadIdx.x; 

   int x = blockIdx.x * blockDim.x + threadIdx.x + column;
   int y = blockIdx.y * blockDim.y + threadIdx.y + row;

   if (x >= resWidth || y >= resHeight) return;

   int cacheIdx = y * resWidth + x;
   Ray ray = cache[cacheIdx].ray;
   glm::vec3 scale = cache[cacheIdx].scale;

   vec3 totalColor(0.0f);
   bool killed = true;
   float t;
   Triangle *closestGeom;
   if (scale.x >= EPSILON || scale.y >= EPSILON || scale.z >= EPSILON) {
      getClosestIntersection(ray, tree, &closestGeom, &t);
      if (closestGeom != kNoShapeFound) {
         totalColor += scale * shadeObject(tree, lights, lightCount, 
               closestGeom, t, ray, &randStates[tid]);

         if (!closestGeom->getMaterial().emissive && depth >= kMinDepth) {
            if (curand_uniform(&randStates[tid]) < kRussianRoulette) {
            } else {
               scale *= 1.0f / kRussianRoulette;
               killed = false;
            }
         } else {
            killed = false;
         }
      }
   } 

   if (!killed) {
      glm::vec3 normal = closestGeom->getNormalAt(ray, t);
      glm::vec3 point = ray.getPoint(t);
      float randNum = curand_uniform(&randStates[tid]); 
      Material m = closestGeom->getMaterial();
      float reflThresh = m.refl;
      float difThresh = 1.0f - m.alpha;

      Ray newRay(vec3(0.0f), vec3(0.0f));
      if (randNum < reflThresh) {
         vec3 eyeVec = glm::normalize(-ray.d);

         newRay = Ray(point, 2.0f * glm::dot(normal, eyeVec) * normal - eyeVec);

      // Shoot an indirect ray
      } else if (randNum < difThresh) {
         float rand1 = curand_uniform(&randStates[tid]); 
         float rand2 = curand_uniform(&randStates[tid]);
         vec3 dir = cosineWeightedSample(normal, rand1, rand2);
         newRay = Ray(point, dir);
         scale = m.dif * m.clr / (1.0f - m.alpha - m.refl);

      // Do refraction
      } else {
         vec3 eyeVec = glm::normalize(-ray.d);
         vec3 refrNorm;
         vec3 d = -eyeVec;
         float n1, n2;

         if (isFloatLessThan(glm::dot(eyeVec, normal), 0.0f)) {
            n1 = m.ior; n2 = kAirIOR;
            refrNorm = -normal;
         } else { 
            n1 = kAirIOR; n2 = m.ior;
            refrNorm = normal;
         }

         float dDotN = glm::dot(d, refrNorm);
         float nr = n1 / n2;
         float discriminant = 1.0f - nr * nr * (1.0f - dDotN * dDotN);
         if (discriminant > 0.0f) {
            vec3 refracDir = nr * (d - refrNorm * dDotN) - refrNorm * sqrtf(discriminant);
            newRay = Ray(point, refracDir);
         }
      }

      newRay.o += newRay.d * BIG_EPSILON;
      cache[cacheIdx] = RayCache(newRay, scale);
   }

   totalColor = vec3(clamp(totalColor.x, 0, 1), 
                         clamp(totalColor.y, 0, 1), 
                         clamp(totalColor.z, 0, 1)); 

   output[x + y * resWidth] += totalColor;
}

void allocateGPUScene(const TKSceneData &data, Triangle **dGeomList,
      Triangle ***g_dLightList, int *retGeometryCount, 
      int *retLightCount, Shader **g_dShader, ShadingType stype) {
   int geometryCount = 0;
   int biggestListSize = 0;
   int lightCount = 0;

   int imgWidth, imgHeight;
   unsigned char *texData = readBMP("blitz.bmp", &imgWidth, &imgHeight);

   int imgSize = sizeof(uchar4) * imgWidth * imgHeight;
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

   cudaArray* cu_array;
   cudaMallocArray(&cu_array, &channelDesc, imgWidth, imgHeight );

   //copy image to device array cu_array – used as texture mytex on device
   HANDLE_ERROR(cudaMemcpyToArray(cu_array, 0, 0, texData, imgSize, cudaMemcpyHostToDevice));
   // set texture parameters
   
   mytex.addressMode[0] = cudaAddressModeWrap;
   mytex.addressMode[1] = cudaAddressModeWrap;
   mytex.filterMode = cudaFilterModeLinear;
   mytex.normalized = true; 

   // Bind the array to the texture
   HANDLE_ERROR(cudaBindTextureToArray(mytex, cu_array, channelDesc));

   TKTriangle *dTriangleTokens = NULL;
   TKSmoothTriangle *dSmthTriTokens = NULL;

   // Count all the triangles that are also lights
   for (int i = 0; i < data.triangles.size(); i++) { 
      if (data.triangles[i].mod.fin.em) lightCount++;
   }
   for (int i = 0; i < data.smoothTriangles.size(); i++) { 
      if (data.smoothTriangles[i].mod.fin.em) lightCount++;
   }

   int triangleCount = data.triangles.size();
   if (triangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dTriangleTokens, sizeof(TKTriangle) * triangleCount));
      HANDLE_ERROR(cudaMemcpy(dTriangleTokens, &data.triangles[0], 
               sizeof(TKTriangle) * triangleCount, cudaMemcpyHostToDevice));
      geometryCount += triangleCount;
      if (triangleCount > biggestListSize) biggestListSize = triangleCount;
   }

   int smoothTriangleCount = data.smoothTriangles.size();
   if (smoothTriangleCount > 0) {
      HANDLE_ERROR(cudaMalloc(&dSmthTriTokens, 
               sizeof(TKSmoothTriangle) * smoothTriangleCount));
      HANDLE_ERROR(cudaMemcpy(dSmthTriTokens, &data.smoothTriangles[0],
               sizeof(TKSmoothTriangle) * smoothTriangleCount, cudaMemcpyHostToDevice));
      geometryCount += smoothTriangleCount;
      if (smoothTriangleCount > biggestListSize) biggestListSize = smoothTriangleCount;
   }

   

   HANDLE_ERROR(cudaMalloc(dGeomList, sizeof(Triangle) * geometryCount));
   HANDLE_ERROR(cudaMalloc(g_dLightList, sizeof(Triangle *) * lightCount));

   int blockSize = kBlockWidth * kBlockWidth;
   int gridSize = (biggestListSize - 1) / blockSize + 1;
   // Fill up GeomList and LightList with actual objects on the GPU
   initScene<<<gridSize, blockSize>>>(*dGeomList, dTriangleTokens, triangleCount, dSmthTriTokens, smoothTriangleCount, 
         stype);

   cudaDeviceSynchronize();
   checkCUDAError("initScene failed");

   if (dTriangleTokens) HANDLE_ERROR(cudaFree(dTriangleTokens));
   if (dSmthTriTokens) HANDLE_ERROR(cudaFree(dSmthTriTokens));

   *retGeometryCount = geometryCount;
   *retLightCount = lightCount;
}

__global__ void averagePasses(int resWidth, int resHeight, vec3 image[], vec3 outImage[], int passes) {
   int x = blockIdx.x * blockDim.x + threadIdx.x; 
   int y = blockIdx.y * blockDim.y + threadIdx.y; 
   if (x >= resWidth || y >= resHeight) return;

   int idx = y * resWidth + x;

   outImage[idx] = image[idx] * 255.0f / (float)passes;
}

__device__ float getMedian(vec3 list[], int listSize, int axis) {
   int pivot;
   int start = 0;
   int end = listSize;
   int topOfBottom;
   do {
      pivot = (start + end) / 2;
      SWAP(list[pivot], list[end - 1]); 
      topOfBottom = start;
      for (int i = start; i < end - 1; i++) {
         if (list[i][axis] < list[pivot][axis]) {
            SWAP(list[i], list[topOfBottom]);
            topOfBottom++;
         }
      }
      SWAP(list[end - 1], list[topOfBottom]);
      if (topOfBottom < listSize / 2) start = topOfBottom + 1;
      else if (topOfBottom > listSize / 2) end = topOfBottom;
   } while (topOfBottom != listSize / 2);
   return list[pivot][axis];
}

__global__ void medianFilter(int resWidth, int resHeight, vec3 image[]) {
   int x = blockIdx.x * blockDim.x + threadIdx.x; 
   int y = blockIdx.y * blockDim.y + threadIdx.y; 

   if (x >= resWidth || y >= resHeight) return;
   int idx = y * resWidth + x;

   vec3 nearClrs[kMedianPixelAmt * kMedianPixelAmt];
   for (int i = 0; i < kMedianPixelAmt; i++) {
      for (int j = 0; j < kMedianPixelAmt; j++) {
         nearClrs[i * kMedianPixelAmt + j] = image[min(max((y + j - kMedianPixelAmt / 2) * resWidth + x + i - kMedianPixelAmt / 2, 0), resWidth * resHeight - 1)];
      }
   }
   
   image[idx].x = getMedian(nearClrs, kMedianPixelAmt * kMedianPixelAmt, kXAxis);
   image[idx].y = getMedian(nearClrs, kMedianPixelAmt * kMedianPixelAmt, kYAxis);
   image[idx].z = getMedian(nearClrs, kMedianPixelAmt * kMedianPixelAmt, kZAxis);
}

__global__ void blurFilter(int resWidth, int resHeight, vec3 image[]) {
   int x = blockIdx.x * blockDim.x + threadIdx.x; 
   int y = blockIdx.y * blockDim.y + threadIdx.y; 

   if (x >= resWidth || y >= resHeight) return;
   int idx = y * resWidth + x;

   vec3 nearClrs[kMedianPixelAmt * kMedianPixelAmt];
   for (int i = 0; i < kMedianPixelAmt; i++) {
      for (int j = 0; j < kMedianPixelAmt; j++) {
         nearClrs[i * kMedianPixelAmt + j] = image[min(max((y + j - kMedianPixelAmt / 2) * resWidth + x + i - kMedianPixelAmt / 2, 0), resWidth * resHeight - 1)];
      }
   }

   vec3 clr(0.0f);
   for (int i = 0; i < kMedianPixelAmt * kMedianPixelAmt; i++) {
      clr += nearClrs[i];
   }
   image[idx] = clr / (float)(kMedianPixelAmt * kMedianPixelAmt);
}

__global__ void convertToUchar4(int resWidth, int resHeight, vec3 vec3Clrs[], uchar4 output[]) {
   
   int x = blockIdx.x * blockDim.x + threadIdx.x; 
   int y = blockIdx.y * blockDim.y + threadIdx.y; 
   // Flip the image
   int flippedY = resHeight - y - 1; 


   if (x >= resWidth || y >= resHeight) return;

   int idx = y * resWidth + x;
   vec3 clr = vec3Clrs[idx];
   uchar4 convClr;
   convClr.x = clamp(clr.x, 0, 255); convClr.y = clamp(clr.y, 0, 255); 
   convClr.z = clamp(clr.z, 0, 255); convClr.w = 255;
   output[flippedY * resWidth + x] = convClr;
}

RayCache *g_dRayCache;
BVHTree *g_dBvhTree;
curandState *g_dRandStates;
vec3 *g_dVec3Out;
vec3 *g_dTempImage;
Shader **g_dShader;
Triangle **g_dLightList;
int g_lightCount;
Camera g_camera(vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f));

extern "C" void init_kernel(const TKSceneData &data, ShadingType stype, int width, int height) {
   Triangle *dGeomList; 
   int geometryCount;

   dim3 dimBlock, dimGrid;

   HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kGimmeLotsOfMemory));

   TKCamera camTK = *data.camera;
   g_camera = Camera(camTK.pos, camTK.up, camTK.right, 
                 glm::normalize(camTK.lookAt - camTK.pos));

   // Fill the geomList and light list with objects dynamically created on the GPU
   HANDLE_ERROR(cudaMalloc(&g_dShader, sizeof(Shader*)));
   HANDLE_ERROR(cudaMalloc(&g_dVec3Out , sizeof(vec3) * width * height));
   HANDLE_ERROR(cudaMalloc(&g_dTempImage, sizeof(vec3) * width * height));
   HANDLE_ERROR(cudaMemset(g_dVec3Out, 0, sizeof(vec3) * width * height));

   allocateGPUScene(data, &dGeomList, &g_dLightList, &geometryCount, &g_lightCount, g_dShader, stype);

   cudaDeviceSynchronize();
   checkCUDAError("AllocateGPUScene failed");

   HANDLE_ERROR(cudaMalloc(&g_dBvhTree, sizeof(BVHTree)));
   formBVH(dGeomList, geometryCount, g_dBvhTree);

   //formBVH modifies the ordering of geometry so the lights must be gathered afterwards
   setupLights<<<1, 1>>>(dGeomList, geometryCount, g_dLightList);

   HANDLE_ERROR(cudaMalloc(&g_dRandStates, sizeof(curandState) * kBlockWidth * kBlockWidth));

   dimBlock = dim3(kBlockWidth * kBlockWidth);
   initCurand<<<1, dimBlock>>>(g_dRandStates, kBlockWidth * kBlockWidth);
   HANDLE_ERROR(cudaMalloc(&g_dRayCache, sizeof(RayCache) * width* height));

}

extern "C" void launch_kernel(int width, int height, int maxDepth, int pass, uchar4 *dOutput, bool blur, bool median) {

   dim3 dimBlock = dim3(kBlockWidth, kBlockWidth);
   dim3 camGrid((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
   dim3 dimGrid = dim3((kColumnsPerKernel - 1) / kBlockWidth + 1, (kColumnsPerKernel - 1) / kBlockWidth + 1);

   generateCameraRays<<<camGrid, dimBlock>>>(width, height, g_camera, g_dRayCache, g_dRandStates);
   if (pass == 1) cudaMemset(g_dVec3Out, 0, sizeof(vec3) * width * height);

   cudaDeviceSynchronize();
   for (int depth = 0; depth < maxDepth; depth++) {
      for (int x = 0; x < width ; x += kColumnsPerKernel) {
         for (int y = 0; y < height; y += kRowsPerKernel) {
            rayTrace<<<dimGrid, dimBlock>>>(x, y, width, height,
               g_dBvhTree, g_dLightList, g_lightCount, g_dVec3Out, 
               g_dRandStates, g_dRayCache, depth);
         }
      }
   }
   cudaDeviceSynchronize();
   checkCUDAError("rayTrace kernel failed");

   dimBlock = dim3(kBlockWidth, kBlockWidth);
   dimGrid = dim3((width - 1) / kBlockWidth + 1, (height - 1) / kBlockWidth + 1);
   averagePasses<<<dimGrid, dimBlock>>>(width, height, g_dVec3Out, g_dTempImage, pass); 
   cudaDeviceSynchronize();
   checkCUDAError("averagePasses kernel failed");

   if (median) {
      medianFilter<<<dimGrid, dimBlock>>>(width, height, g_dTempImage); 
      cudaDeviceSynchronize();
      checkCUDAError("medianFilter kernel failed");
   }

   if (blur) {
      blurFilter<<<dimGrid, dimBlock>>>(width, height, g_dTempImage); 
      cudaDeviceSynchronize();
      checkCUDAError("blurFilter kernel failed");
   }

   convertToUchar4<<<dimGrid, dimBlock>>>(width, height, g_dTempImage, dOutput); 
   cudaDeviceSynchronize();
   checkCUDAError("convertToUchar4 kernel failed");
}

void translateCamera(glm::vec3 dir) {
   g_camera.pos += dir.x * glm::normalize(g_camera.right);
   g_camera.pos += dir.z * g_camera.lookAtDir;
   g_camera.pos += dir.y * g_camera.up;
}

void rotateCameraSideways(float angle) {
   mat4 rot = glm::rotate(mat4(1.0f), angle, g_camera.up);
   g_camera.right = vec3(rot * vec4(g_camera.right.x, 
                                    g_camera.right.y, 
                                    g_camera.right.z, 0.0f));
   g_camera.lookAtDir = vec3(rot * vec4(g_camera.lookAtDir.x, 
                                    g_camera.lookAtDir.y, 
                                    g_camera.lookAtDir.z, 0.0f));
}
void rotateCameraVertically(float angle) {
   glm::mat4 rot = glm::rotate(glm::mat4(1.0f), angle, g_camera.right);
   g_camera.up = vec3(rot * vec4(g_camera.up.x, 
                                    g_camera.up.y, 
                                    g_camera.up.z, 0.0f));
   g_camera.lookAtDir = vec3(rot * vec4(g_camera.lookAtDir.x, 
                                    g_camera.lookAtDir.y, 
                                    g_camera.lookAtDir.z, 0.0f));
}

void kernelGetImage(uchar4 *dImage, uchar4 *hostImage, int width, int height) {
   HANDLE_ERROR(cudaMemcpy(hostImage, dImage, 
            sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
   cudaDeviceSynchronize();
}
