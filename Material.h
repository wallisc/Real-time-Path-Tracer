#ifndef MATERIAL_H
#define MATERIAL_H

#include "glm/glm.hpp"

const int NO_TEXTURE = -1;

typedef struct Material {
   glm::vec3 clr;
   float alpha, amb, dif, spec, rough, refl, refr, ior;
   bool emissive;
   int texId;

   __device__ Material(const glm::vec3 &lightColor) : clr(lightColor), emissive(true) {}
   __device__ Material(const glm::vec3 &color, float alphaVal, float ambiant, 
        float diffuse, float specular, float roughness, float reflection, 
        float refraction, float indexOfRefraction, int textureId = NO_TEXTURE) :
     clr(color), amb(ambiant), dif(diffuse), spec(specular), rough(roughness), 
     refl(reflection), refr(refraction), ior(indexOfRefraction), alpha(alphaVal),
     texId(textureId), emissive(false) {} 

} Material;
#endif //MATERIAL_H
