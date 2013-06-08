#include "bvh.h"
#include "kernel.h"

using glm::vec3;
using std::vector;
using std::pair;

typedef pair<int, int> SortRange;

BVHNode **g_dBVHLeaves;
BVHNode **g_dBVHbuffer1;
BVHNode **g_dBVHbuffer2;
int g_numLeaves;

void multiKernelSort(Triangle buffer1[], Triangle buffer2[], 
      const vector<SortRange > &sortIdxs, int axis);

__global__ void merge(Triangle oldBuffer[], Triangle newBuffer[], int width, 
      int size, int axis);

__global__ void copyOver(Triangle writeTo[], Triangle readFrom[], int size);

__global__ void convergeBVHNodes(Triangle oldBuffer[], Triangle newBuffer[], 
      int bufferSize, int nodesLeft);

__global__ void createBVHTree(BVHTree *tree, BVHNode *nodes[]);

__global__ void setupBVHNodes(Triangle geomList[], int geomCount, BVHNode *nodeBuffer[]);

__global__ void convergeBVHNodes(BVHNode *oldBuffer[], BVHNode *newBuffer[], 
      int bufferSize, int nodesLeft);


void formBVH(Triangle dGeomList[], int geomCount, BVHTree *dTree) {

   g_numLeaves = geomCount;
   // Sort all the geometry data in a way that makes creating the BVH tree
   // parallelizable 
   vector<SortRange > *oldQueue = new vector<SortRange >();
   vector<SortRange > *newQueue = new vector<SortRange >();

   Triangle *dMergeSortBuffer;
   HANDLE_ERROR(cudaMalloc(&dMergeSortBuffer, sizeof(Triangle) * geomCount));

   int start = 0, end;
   int axis = kXAxis;
   oldQueue->push_back(SortRange(0, geomCount));
   while(oldQueue->size() > 0) {

      // Sort everything in the queue
      multiKernelSort(dGeomList, dMergeSortBuffer, *oldQueue, axis);

      // Figure out what the next batch of sorts needs to be
      while (oldQueue->size() > 0) {
         start = oldQueue->back().first;
         end = oldQueue->back().second;
         oldQueue->pop_back();

         if (end - start > 2) {

            int closestPow2 = 2;
            while (closestPow2 * 2 < end - start) closestPow2 *= 2;

            newQueue->push_back(SortRange(start, closestPow2));
            newQueue->push_back(SortRange(start + closestPow2, end));
         }       
      }

      SWAP(newQueue, oldQueue);
      axis = (axis + 1) % kAxisNum;
   }

   // Create the BVH
   int bufferSize = (geomCount - 1) / 2 + 1;

   HANDLE_ERROR(cudaMalloc(&g_dBVHLeaves, sizeof(BVHNode *) * bufferSize));
   HANDLE_ERROR(cudaMalloc(&g_dBVHbuffer1, sizeof(BVHNode *) * bufferSize));
   HANDLE_ERROR(cudaMalloc(&g_dBVHbuffer2, sizeof(BVHNode *) * bufferSize));

   int blockSize = kBlockWidth * kBlockWidth;
   int gridSize = (bufferSize - 1) / blockSize + 1;
   setupBVHNodes<<<gridSize, blockSize>>>(dGeomList, geomCount, g_dBVHLeaves);
   cudaDeviceSynchronize();
   checkCUDAError("setupBVHNodes failed");
   HANDLE_ERROR(cudaMemcpy(g_dBVHbuffer1, g_dBVHLeaves, sizeof(BVHNode *) * bufferSize, cudaMemcpyDeviceToDevice));

   // For every level of the BVH tree (from the bottom up)
   for(int nodesLeft = bufferSize; nodesLeft > 1; nodesLeft = (nodesLeft - 1) / 2 + 1) {
      gridSize = (nodesLeft - 1) / blockSize + 1;
      convergeBVHNodes<<<gridSize, blockSize>>>(g_dBVHbuffer1, g_dBVHbuffer2, bufferSize, nodesLeft);
      cudaDeviceSynchronize();
      checkCUDAError("convergeBVHNodes failed");
      SWAP(g_dBVHbuffer1, g_dBVHbuffer2);
   }

   createBVHTree<<<1, 1>>>(dTree, g_dBVHbuffer1);
   cudaDeviceSynchronize();
   checkCUDAError("createBVHTree failed");
   
   HANDLE_ERROR(cudaFree(dMergeSortBuffer));
}

// Given a batch of 'ranges' to sort, runs multiple mergesorts concurrently
void multiKernelSort(Triangle buffer1[], Triangle buffer2[], 
      const vector<SortRange> &sortIdxs, int axis) {

   int blockSize = kBlockWidth * kBlockWidth;
   int size = 0;
   int swaps = 1;

   // Find the longest sort range
   for (int i = 0; i < sortIdxs.size(); i++) {
      int sortSize = sortIdxs[i].second - sortIdxs[i].first;
      if (sortSize > size) size = sortSize;
   }

   for (int width = 1; width < size + 1; width = 2 * width) {
      for (int i = 0; i < sortIdxs.size(); i++) {
         int start = sortIdxs[i].first;
         int end = sortIdxs[i].second;
         int sortSize = end - start;

         if (width >= sortSize) continue;

         int divs = (sortSize - 1) / (width * 2) + 1;
         int gridSize = (divs - 1) / blockSize + 1; 
         merge<<<gridSize, blockSize>>>(buffer1 + start, 
               buffer2 + start, width, sortSize, axis);

         // If the completely sorted data is in buffer2, copy it back over to 
         // buffer1
         if (divs == 1 && swaps % 2 == 1) {
            gridSize = (sortSize - 1) / blockSize + 1;
            copyOver<<<gridSize, blockSize>>>
                  (buffer1 + start, buffer2 + start, sortSize);
         }
      }
      swaps++;
      SWAP(buffer1, buffer2);
      cudaDeviceSynchronize();
      checkCUDAError("mergeSort failed");
   }
}

// oldBuffer - The array containing the items to be sorted
// newBuffer - Array to copy items to
// width - width of each 'division' to be merged
// size - size of the buffer
// axis - axis to sort on
__global__ void merge(Triangle oldBuffer[], Triangle newBuffer[], int width, 
      int size, int axis) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int i1 = idx * width * 2;
   int i2 = idx * width * 2 + width;
   int div1End = min(i2, size);
   int div2End = min(i2 + width, size); 

   for(int i = i1; i < div2End; i++) {
      if (i1 < div1End && (i2 >= div2End || 
         oldBuffer[i1].getCenter()[axis] < oldBuffer[i2].getCenter()[axis])) {

         newBuffer[i] = oldBuffer[i1++];
      } else {
         newBuffer[i] = oldBuffer[i2++];
      }
   }
}

__global__ void copyOver(Triangle writeTo[], Triangle readFrom[], int size) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= size) return;
   writeTo[idx] = readFrom[idx];
}

__global__ void setupBVHNodes(Triangle geomList[], int geomCount, BVHNode *nodeBuffer[]) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx * 2 >= geomCount) return;

   if (idx * 2 + 1 < geomCount) {
      nodeBuffer[idx] = new BVHNode();
      
      nodeBuffer[idx]->left = new BVHNode(&geomList[idx * 2]);
      nodeBuffer[idx]->right = new BVHNode(&geomList[idx * 2 + 1]);
      nodeBuffer[idx]->right->parent = nodeBuffer[idx];
      nodeBuffer[idx]->left->parent = nodeBuffer[idx];
      nodeBuffer[idx]->bb = combineBoundingBox(nodeBuffer[idx]->left->bb, 
                                               nodeBuffer[idx]->right->bb);
   } else {
      nodeBuffer[idx] = new BVHNode(&geomList[idx * 2]);       
   } 
}

__global__ void updateLeaves(BVHNode *leaves[], float dt, int numLeaves) {
   int uid = blockIdx.x * blockDim.x + threadIdx.x;
   BVHNode *leaf;
   int leafIdx = uid / 2;

   if (uid >= numLeaves) return;

   if (leaves[leafIdx]->geom) leaf = leaves[leafIdx];
   else if (uid % 2 == 0) leaf = leaves[leafIdx]->left;
   else leaf = leaves[leafIdx]->right;
   leaf->geom->update(dt); 
   leaf->bb = leaf->geom->getBoundingBox();
}

__global__ void updateBVHNodes(BVHNode *oldBuffer[], BVHNode *newBuffer[], int nodesLeft) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (idx >= nodesLeft) return;

   if (idx == nodesLeft - 1 && nodesLeft % 2 == 1) {
      newBuffer[idx / 2] = oldBuffer[idx];
   } else {
      oldBuffer[idx]->bb = combineBoundingBox(oldBuffer[idx]->left->bb, 
                                              oldBuffer[idx]->right->bb);
      newBuffer[idx / 2] = oldBuffer[idx]->parent;
   }

}
void updateBVH(float dt) {
   int blockSize = kBlockWidth * kBlockWidth;
   int gridSize = (g_numLeaves - 1) / blockSize + 1;

   updateLeaves<<<gridSize, blockSize>>>(g_dBVHLeaves, dt, g_numLeaves);
   cudaDeviceSynchronize();
   checkCUDAError("updateLeaves kernel failed");
  
   int bufferSize = (g_numLeaves - 1) / 2 + 1;
   HANDLE_ERROR(cudaMemcpy(g_dBVHbuffer1, g_dBVHLeaves, sizeof(BVHNode *) * bufferSize, cudaMemcpyDeviceToDevice));
   for(int nodesLeft = bufferSize; ; nodesLeft = (nodesLeft - 1) / 2 + 1) {
      gridSize = (nodesLeft - 1) / blockSize + 1;
      updateBVHNodes<<<gridSize, blockSize>>>(g_dBVHbuffer1, g_dBVHbuffer2, nodesLeft);
      cudaDeviceSynchronize();
      checkCUDAError("updateBVHNodes failed");
      SWAP(g_dBVHbuffer1, g_dBVHbuffer2);
      if (nodesLeft == 1) break;
   }
}

__global__ void convergeBVHNodes(BVHNode *oldBuffer[], BVHNode *newBuffer[], 
      int bufferSize, int nodesLeft) {

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (idx >= bufferSize) return;
   newBuffer[idx] = NULL;

   if (idx * 2 >= bufferSize) return;

   if (idx * 2 + 1 < nodesLeft) {
      newBuffer[idx] = new BVHNode();
      oldBuffer[idx * 2]->parent = newBuffer[idx];
      oldBuffer[idx * 2 + 1]->parent = newBuffer[idx];
      newBuffer[idx]->left = oldBuffer[idx * 2];
      newBuffer[idx]->right = oldBuffer[idx * 2 + 1];
      newBuffer[idx]->bb = combineBoundingBox(newBuffer[idx]->left->bb, 
                                              newBuffer[idx]->right->bb);
      BoundingBox bb = newBuffer[idx]->bb;
   } else if (idx * 2 < nodesLeft) {
      newBuffer[idx] = oldBuffer[idx * 2];       
   } 
}


__global__ void createBVHTree(BVHTree *tree, BVHNode *nodes[]) {
   tree->root = nodes[0];
}

