#ifndef BVH_H
#define BVH_H
#include "Triangle.h"
#include "BoundingBox.h"
#include "GeometryUtil.h"
#include "Util.h"
#include "kernel.h"

typedef struct BVHNode {
   BVHNode *left, *right;
   Triangle *geom;
   BoundingBox bb;

   __device__ BVHNode() : left(NULL), right(NULL), geom(NULL) {}
   __device__ BVHNode(Triangle *object) : left(NULL), right(NULL), geom(object) {
      bb = geom->getBoundingBox();
   }
} BVHNode;

typedef struct BVHTree {
   BVHNode *root;
} BVHTree;

void formBVH(Triangle *dGeomList[], int geomCount, BVHTree *dTree);

#endif //BVH_H
