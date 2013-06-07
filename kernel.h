#ifndef KERNEL_H
#define KERNEL_H
#include <stdio.h>
#include <cuda.h>
#include <vector_types.h>

#include "curand_kernel.h"
#include "TokenData.h"
#include "Geometry.h"
#include "PointLight.h"
#include "Shader.h"
#include "bvh.h"

const int kMaxStackSize = 70;
const int kGimmeLotsOfMemory = 1000000 * 256;
const int kBlockWidth = 16;
const int kNumStreams = 3;
const int kMaxTextures= 10;

typedef enum {PHONG, COOK_TORRANCE} ShadingType;
const float kAirIOR = 1.0f;

const int kMedianPixelAmt = 3;

const int kMinDepth= 3;
const float kRussianRoulette= .3f;

const int kXAxis = 0, kYAxis = 1, kZAxis = 2;
const int kAxisNum = 3;

const int kMaxLights = 10;

const int kRowsPerKernel = 512;
const int kColumnsPerKernel = 512;

extern "C" void init_kernel(const TKSceneData &data, ShadingType stype, int width, int height);
extern "C" void launch_kernel(int width, int height, int maxDepth, int pass, uchar4 *output, bool blur, bool median);
void kernelGetImage(uchar4 *dImage, uchar4 *image, int width, int height);
void translateCamera(glm::vec3 trans);
void rotateCameraVertically(float angle);
void rotateCameraSideways(float angle);

typedef struct RayCache {
   Ray ray;
   glm::vec3 scale;
   __device__ RayCache(const Ray &nRay, glm::vec3 nScale) : ray(nRay), scale(nScale) {} 
} RayCache;

#endif //KERNEL_H
