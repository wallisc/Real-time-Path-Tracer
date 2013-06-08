#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Material.h"
#include "glm/glm.hpp"
#include "Util.h"

class Triangle {
public:
   __device__ Triangle(const glm::vec3 &point1, const glm::vec3 &point2, 
         const glm::vec3 &point3, const glm::vec3 &norm1, 
         const glm::vec3 &norm2, const glm::vec3 &norm3, const Material &nMat, 
         const glm::vec3 velocity, glm::vec2 nVt1 = glm::vec2(0.0f), glm::vec2 nVt2 = glm::vec2(0.0f), 
         glm::vec2 nVt3 = glm::vec2(0.0f)) : p1(point1), p2(point2), p3(point3), 
                                             n1(norm1), n2(norm2), n3(norm3), vel(velocity),
                                             vt1(nVt1), vt2(nVt2), vt3(nVt3), mat(nMat) {
      n = glm::normalize(glm::cross(point2 - point1, point3 - point1));
      c = point1;
   }

   __device__ Material getMaterial() const { return mat; };

   // Precondition: the ray intersects with the triangle
   __device__ glm::vec3 getNormalAt(const Ray &ray, float param) const {
      glm::vec3 q = ray.getPoint(param);
      float area = glm::dot(glm::cross(p2 - p1, p3 - p1), n);
      float beta = glm::dot(glm::cross(p1 - p3, q - p3), n) / area;
      float gamma = glm::dot(glm::cross(p2 - p1, q - p1), n) / area;
      float alpha = 1.0 - beta - gamma;
      return glm::normalize(alpha * n1 + beta * n2 + gamma * n3);
   }

   __device__ glm::vec3 getCenter() const { 
      return (p1 + p2 + p3) / 3.0f;
   }

   __device__ BoundingBox getBoundingBox() const {
      glm::vec3 min = getSmallestBoxCorner(getSmallestBoxCorner(p1, p2), p3);
      glm::vec3 max = getBiggestBoxCorner(getBiggestBoxCorner(p1, p2), p3);

      return BoundingBox(min, max);
   }

   __device__ glm::vec2 UVAt(const Ray &r, float param) const {
      glm::vec3 q = r.getPoint(param);
      float area = glm::dot(glm::cross(p2 - p1, p3 - p1), n);
      float beta = glm::dot(glm::cross(p1 - p3, q - p3), n) / area;
      float gamma = glm::dot(glm::cross(p2 - p1, q - p1), n) / area;
      float alpha = 1.0 - beta - gamma;
      
      return alpha * vt1 + beta * vt2 + gamma * vt3;
   }

   __device__ glm::vec3 getPointFromBary(float alpha, float beta) {
      float gamma = 1.0f - alpha - beta;
      return p1 * alpha + p2 * beta + p3 * gamma; 
   }

   __device__ float getIntersection(const Ray &r) const {
      float numer = glm::dot(-n,r.o - c);
      float denom = glm::dot(n, r.d);
      float t;

      if (isFloatZero(numer) || isFloatZero(denom) || 
            !isFloatAboveZero(t = numer / denom))
         return -1.0f;

      // intersection point
      glm::vec3 P = r.getPoint(t);

      glm::vec3 A = p1, B = p2, C = p3;
      glm::vec3 AB = B - A;
      glm::vec3 AC = C - A;

      glm::vec3 AP = P - A;
      glm::vec3 ABxAP = glm::cross(AB, AP);
      float v_num = glm::dot(n, ABxAP);
      if (v_num < 0.0f) return -1.0f;

      
      // edge 2
      glm::vec3 BP = P - B;
      glm::vec3 BC = C - B;
      glm::vec3 BCxBP = glm::cross(BC, BP);
      if (glm::dot(n, BCxBP) < 0.0f)
         return -1.0f; // P is on the left side

      // edge 3, needed to compute u
      glm::vec3 CP = P - C;
      // we have computed AC already so we can avoid computing CA by
      // inverting the vectors in the cross product:
      // Cross(CA, CP) = cross(CP, AC);
      glm::vec3 CAxCP = glm::cross(CP, AC);
      float u_num = glm::dot(n, CAxCP);
      if (u_num < 0.0f)
         return -1.0f; // P is on the left side;

      return t;
   }

   __device__ void update(float dt) {
      glm::vec3 dX = dt * vel; // Change in position
      c += dX;
      p1 += dX;
      p2 += dX;
      p3 += dX;
   }

   Material mat;
   // Normal and center of the plane the triangle is sitting on
   // TODO c is just p1, kept for syntax readability
   glm::vec3 vel;
   glm::vec3 n, c;
   glm::vec3 p1, p2, p3;
   glm::vec3 n1, n2, n3;
   glm::vec2 vt1, vt2, vt3;
};

#endif //TRIANGLE_H
