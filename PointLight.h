#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "Light.h"

class PointLight {
public:
   __device__ glm::vec3 getLightAtPoint(glm::vec3 point) const {
      return c;
   }

   __device__ glm::vec3 getLightDir(glm::vec3 point) const {
      return glm::normalize(p - point);
   }

   __device__ Ray getShadowFeeler(glm::vec3 point) const {
      return Ray(p, point - p);
   }

   glm::vec3 p, c;
};

#endif //POINT_LIGHT_H
