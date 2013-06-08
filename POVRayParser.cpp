#include "POVRayParser.h"
#include <sstream>
#include <stdio.h>
#include "glm/glm.hpp"
#include "glm/core/func_matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"

using namespace std;
using glm::vec3;
using glm::vec4;
using glm::mat4;

int POVRayParser::parseFile(const string &fileName, TKSceneData *data) {
   ifstream in(fileName.c_str());
   char strBuffer[kBufferSize];
   int status;

   if (!in.good()) {
      cerr << "File \"" + fileName + "\" not found" << endl;
      return kFileNotFound;
   }

   while (!in.eof()) {

      checkComments(in);

      string firstWord;
      in >> firstWord;

      if (firstWord.length() == 0) {
         continue;
      } else if (!firstWord.compare("light_source")) {
         status = parseLightSource(in, data);
      } else if (!firstWord.compare("area_light")) {
         status = parseAreaLight(in, data);
      } else if (!firstWord.compare("camera")) {
         status = parseCamera(in, data);
      } else if (!firstWord.compare("box")) {
         status = parseBox(in, data);
      } else if (!firstWord.compare("sphere")) {
         status = parseSphere(in, data);
      } else if (!firstWord.compare("cone")) {
         status = parseCone(in, data);
      } else if (!firstWord.compare("plane")) {
         status = parsePlane(in, data);
      } else if (!firstWord.compare("triangle")) {
         status = parseTriangle(in, data);
      } else if (!firstWord.compare("smooth_triangle")) {
         status = parseSmoothTriangle(in, data);
      } else {
         cerr << "Unrecognized object: ";
         cerr << firstWord << endl;
         return kBadFormat;
      }
   
      if (status != kSuccess) return status;
   }

   in.close();
   return kSuccess;
}

int POVRayParser::parseLightSource(std::ifstream &in, TKSceneData *data) {
   int numArgs;
   float pX, pY, pZ; 
   float cR, cG, cB;
   char buffer[kBufferSize];

   in.getline(buffer, kBufferSize, '}');
   numArgs = sscanf(buffer, " { < %f , %f , %f > color rgb < %f , %f , %f > ", 
         &pX, &pY, &pZ, &cR, &cG, &cB);

   if (numArgs != 6) {
      cerr << "Invalid format for light_source" << endl;
      return kBadFormat;
   }

   data->pointLights.push_back(TKPointLight(vec3(pX, pY, pZ), vec3(cR, cG, cB)));
#ifdef DEBUG
   printf("Light source is at %f, %f, %f with color %f, %f, %f\n", pX, pY, pZ, cR, cG, cB);
#endif 
   return kSuccess;
}

int POVRayParser::parseAreaLight(std::ifstream &in, TKSceneData *data) {
   int numArgs;
   vec3 v1, v2, v3;
   int samples;
   float cR, cG, cB;
   char buffer[kBufferSize];

   in.getline(buffer, kBufferSize, '}');
   numArgs = sscanf(buffer, " { < %f , %f , %f > < %f , %f , %f > < %f , %f , %f > sample %d color rgb < %f , %f , %f > ", 
         &v1.x, &v1.y, &v1.z, &v2.x, &v2.y, &v2.z, &v3.x, &v3.y, &v3.z, &samples, &cR, &cG, &cB);

   if (numArgs != 13) {
      cerr << "Invalid format for area_light" << endl;
      return kBadFormat;
   }

   cR /= samples * samples;
   cG /= samples * samples;
   cB /= samples * samples;

   vec3 edge1 = v3 - v2;
   vec3 edge2 = v1 - v2;
   for (int x = 0; x < samples; x++) {
      for (int y = 0; y < samples; y++) {
         vec3 loc = v2 + edge1 * (x / (float)samples) + edge2 * (y / (float)samples); 
         data->pointLights.push_back(TKPointLight(loc, vec3(cR, cG, cB)));
      }
   }


#ifdef DEBUG
   printf("Area light is at (%f, %f, %f), (%f, %f, %f), and (%f, %f, %f) with color %f, %f, %f\n", 
         v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, cR, cG, cB);
#endif 
   return kSuccess;
}

int POVRayParser::parseCamera(std::ifstream &in, TKSceneData *data) {
   int numArgs;
   vec3 pos, up, right, lookAt;
   char buffer[kBufferSize];
   string nextWord;
   int status;

   status = parseCharacter(in, '{');   
   if (status != kSuccess) return status;

   in >> nextWord;
   while (nextWord.compare("}")) {

      in.getline(buffer, kBufferSize, '>');
      if(!nextWord.compare("location")) {
         numArgs = sscanf(buffer, " < %f , %f , %f ", &pos.x, &pos.y, &pos.z);
      } else if(!nextWord.compare("up")) {
         numArgs = sscanf(buffer, " < %f , %f , %f ", &up.x, &up.y, &up.z);
      } else if(!nextWord.compare("right")) {
         numArgs = sscanf(buffer, " < %f , %f , %f ", &right.x, &right.y, &right.z);
      } else if(!nextWord.compare("look_at")) {
         numArgs = sscanf(buffer, " < %f , %f , %f ", &lookAt.x, &lookAt.y, &lookAt.z);
      } 

      // Because all fields for a camera are vectors, 
      // all will be expected to have 3 parameters
      if (numArgs != 3) {
         cerr << "Invalid format for camera" << endl;
         return kBadFormat;
      }
      numArgs = 0;

      checkComments(in);   
      in >> nextWord;
   }

#ifdef DEBUG
   printf("Location: %f, %f, %f; Up: %f, %f, %f\n", 
         pos.x, pos.y, pos.z, up.x, up.y, up.z);
   printf("Right: %f, %f, %f; Look_at: %f, %f, %f\n", 
         right.x, right.y, right.z, lookAt.x, lookAt.y, lookAt.z);
#endif 
   data->camera = new TKCamera(pos, up, right, lookAt);
   return kSuccess;
}

int POVRayParser::parseBox(std::ifstream &in, TKSceneData *data) {
   int status;
   TKBox b;

   string nextWord;
   
   status = parseCharacter(in, '{');
   if (status != kSuccess) return status;

   status = parseVector(in, &b.p1);
   if (status != kSuccess) return status;

   in >> nextWord;
   if (nextWord.compare(",")) {
      cerr << "Expected \",\" while parsing for box, found: " << nextWord
           << endl;
      return kBadFormat;
   }

   status = parseVector(in, &b.p2);
   if (status != kSuccess) return status;

#ifdef DEBUG
   printf("Point 1: %f, %f, %f; Point2: %f, %f, %f\n", 
         b.p1.x, b.p1.y, b.p1.z, b.p2.x, b.p2.y, b.p2.z);
#endif 

   status = parseModifiers(in, &b.mod, data);
   data->boxes.push_back(b);
   return status;
}

int POVRayParser::parseSphere(std::ifstream &in, TKSceneData *data) {
   int status;
   TKSphere s;

   string nextWord;
   
   parseCharacter(in, '{');   
   if (status != kSuccess) return status;

   status = parseVector(in, &s.p);
   if (status != kSuccess) return status;

   parseCharacter(in, ',');   
   if (status != kSuccess) return status;

   in >> s.r;

#ifdef DEBUG
   printf("Center: %f, %f, %f; radius: %f\n", 
         s.p.x, s.p.y, s.p.z, s.r);
#endif 

   status = parseModifiers(in, &s.mod, data);

   data->spheres.push_back(s);
   return status;
}

int POVRayParser::parseCone(std::ifstream &in, TKSceneData *data) {
   //cerr << "Cones are currently unsupported" << endl;
   //return kUnsupportedObject;
   int status;
   TKCone c;

   string nextWord;

   status = parseCharacter(in, '{');   
   if (status != kSuccess) return status;

   status = parseVector(in, &c.p1);
   if (status != kSuccess) return status;

   status = parseWord(in, ",");
   if (status != kSuccess) return status;

   in >> c.r1;

   status = parseWord(in, ",");
   if (status != kSuccess) return status;

   status = parseVector(in, &c.p2);
   if (status != kSuccess) return status;

   status = parseWord(in, ",");
   if (status != kSuccess) return status;

   in >> c.r2;

#ifdef DEBUG
   printf("Point 1: %f, %f, %f; Point2: %f, %f, %f\n", 
         c.p1.x, c.p1.y, c.p1.z, c.p2.x, c.p2.y, c.p2.z);
   printf("Radius 1: %f, Radius 2: %f\n", 
         c.r1, c.r2);
#endif 

   status = parseModifiers(in, &c.mod, data);
   return status;
}

int POVRayParser::parsePlane(std::ifstream &in, TKSceneData *data) {
   int status;
   TKPlane p;
   string nextWord;

   status = parseCharacter(in, '{');
   if (status != kSuccess) return status;

   if (!in.good())
      cerr << "Bad stuff happened" << endl;

   status = parseVector(in, &p.n);
   if (status != kSuccess) return status;

   status = parseWord(in, ",");
   if (status != kSuccess) return status;

   in >> p.d;

#ifdef DEBUG
   printf("Normal: %f, %f, %f; Distance: %f\n", 
         p.n.x, p.n.y, p.n.z, p.d);
#endif 

   status = parseModifiers(in, &p.mod, data);

   data->planes.push_back(p);

   return status;
}

int POVRayParser::parseSmoothTriangle(std::ifstream &in, TKSceneData *data) {
   int status;
   TKSmoothTriangle t;
   string nextWord;

   status = parseCharacter(in, '{');   
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p1);
   if (status != kSuccess) return status;
   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.n1);
   if (status != kSuccess) return status;
   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p2);
   if (status != kSuccess) return status;
   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.n2);
   if (status != kSuccess) return status;
   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p3);
   if (status != kSuccess) return status;
   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.n3);
   if (status != kSuccess) return status;

#ifdef DEBUG
   printf("Point 1: %f, %f, %f; Point 2: %f, %f, %f; Point 3: %f, %f, %f\n", 
         t.p1.x, t.p1.y, t.p1.z, t.p2.x, t.p2.y, t.p2.z, t.p3.x, t.p3.y, t.p3.z);
   printf("Normal 1: %f, %f, %f; Normal 2: %f, %f, %f; Normal 3: %f, %f, %f\n", 
         t.n1.x, t.n1.y, t.n1.z, t.n2.x, t.n2.y, t.n2.z, t.n3.x, t.n3.y, t.n3.z);
#endif 

   if (isUV(in)) {
      status = parseWord(in, "uv"); 
      if (status != kSuccess) return status;

      status = parseCharacter(in, '{');   
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt1);
      if (status != kSuccess) return status;

      status = parseCharacter(in, ',');
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt2);
      if (status != kSuccess) return status;

      status = parseCharacter(in, ',');
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt3);
      if (status != kSuccess) return status;

      status = parseCharacter(in, '}');   
      if (status != kSuccess) return status;
   }

   status = parseModifiers(in, &t.mod, data);
   data->smoothTriangles.push_back(t);
   return status;
}

int POVRayParser::parseTriangle(std::ifstream &in, TKSceneData *data) {
   int status;
   TKTriangle t;
   string nextWord;

   status = parseCharacter(in, '{');   
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p1);
   if (status != kSuccess) return status;

   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p2);
   if (status != kSuccess) return status;

   status = parseCharacter(in, ',');
   if (status != kSuccess) return status;

   status = parseVector(in, &t.p3);
   if (status != kSuccess) return status;

   t.n = glm::normalize(glm::cross(t.p2 - t.p1, t.p3 - t.p1));

#ifdef DEBUG
   printf("Point 1: %f, %f, %f; Point 2: %f, %f, %f; Point 3: %f, %f, %f\n", 
         t.p1.x, t.p1.y, t.p1.z, t.p2.x, t.p2.y, t.p2.z, t.p3.x, t.p3.y, t.p3.z);
#endif 
   if (isUV(in)) {
      status = parseWord(in, "uv"); 
      if (status != kSuccess) return status;

      status = parseCharacter(in, '{');   
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt1);
      if (status != kSuccess) return status;

      status = parseCharacter(in, ',');
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt2);
      if (status != kSuccess) return status;

      status = parseCharacter(in, ',');
      if (status != kSuccess) return status;

      status = parseUV(in, &t.vt3);
      if (status != kSuccess) return status;

      status = parseCharacter(in, '}');   
      if (status != kSuccess) return status;
   }

   status = parseModifiers(in, &t.mod, data);
   data->triangles.push_back(t);
   return status;
}

int POVRayParser::parseModifiers(std::ifstream &in, TKModifier *m, TKSceneData *data) {
   string nextWord;
   in >> nextWord;
   int status;
   glm::mat4 matStack(1.0f);
   m->velocity = vec3(0.0f);
    
      while (nextWord.compare("}")) {
         if (!nextWord.compare("scale")) {
            status = parseScale(in, &matStack);
         } else if (!nextWord.compare("rotate")) {
            status = parseRotate(in, &matStack);
         } else if (!nextWord.compare("velocity")) {
            status = parseVector(in, &m->velocity);
         } else if (!nextWord.compare("translate")) {
            status = parseTranslate(in, &matStack);
         } else if (!nextWord.compare("finish")) {
            status = parseFinish(in, &m->fin);
         } else if (!nextWord.compare("pigment")) {
            status = parsePigment(in, &m->pig, data);
         } else {
            cerr << "Invalid format for modifiers\n";
            return kBadFormat;
         }

         if (status != kSuccess) return status;

      in >> nextWord;
   }
   
   m->trans = matStack;
   m->invTrans = glm::inverse(matStack);

   return kSuccess;
}

int POVRayParser::parsePigment(std::ifstream &in, TKPigment *pigment, TKSceneData *data) {
   char buffer[kBufferSize];
   char fileName[kBufferSize];
   vec3 color; 
   float f;   
   int numArgs;

   in.getline(buffer, kBufferSize, '}');
   numArgs = sscanf(buffer, " { color rgb < %f , %f , %f > ",
         &color.x, &color.y, &color.z);

   if (numArgs == 3) {
      *pigment = TKPigment(color);
#ifdef DEBUG
      printf("Pigment color: %f, %f, %f\n", 
            color.x, color.y, color.z);
#endif 
      return kSuccess;
   }

   numArgs = sscanf(buffer, " { color rgbf < %f , %f , %f , %f > ",
         &color.x, &color.y, &color.z, &f);

   if (numArgs == 4) {
      *pigment = TKPigment(color, f);
      return kSuccess;
   } 

   numArgs = sscanf(buffer, " { image_map \"%s\" ", fileName);
   
   if (numArgs == 1) {
      int texId;
      if (!data->textureMap.count(fileName)) {
         data->textureMap[fileName] = texId = data->textureMap.size();
      } else {
         texId = data->textureMap[fileName];
      }
      *pigment = TKPigment(texId);
      return kSuccess;
   } else {
      cerr << "Bad format found for pigment" << endl;
      return kBadFormat;
   }
   
}

// Precondition: Assume finish in not NULL and is filled with default values
int POVRayParser::parseFinish(std::ifstream &in, TKFinish *finish) {
   char buffer[kBufferSize];
    
   string nextWord;
   in >> nextWord;

   if (nextWord[0] != '{') {
      cerr << "Expected \"{\" while parsing finish, found: " << nextWord
           << endl;
      return kBadFormat;
   }
   nextWord = nextWord.substr(1, nextWord.length());

   while(nextWord[0] != '}') {
      if (nextWord.length() == 0) {
         //Do nothing
      } else if (!nextWord.compare("ambient")) {
         in >> finish->amb;
#ifdef DEBUG
         printf("amb: %f\n", finish->amb);
#endif 
      } else if (!nextWord.compare("diffuse")) {
         in >> finish->dif;
#ifdef DEBUG
         printf("dif: %f\n", finish->dif);
#endif 
      } else if (!nextWord.compare("specular")) {
         in >> finish->spec;
#ifdef DEBUG
         printf("spec: %f\n", finish->spec);
#endif 
      } else if (!nextWord.compare("roughness")) {
         in >> finish->rough;
#ifdef DEBUG
         printf("refr: %f\n", finish->rough);
#endif 
      } else if (!nextWord.compare("reflection")) {
         in >> finish->refl;
#ifdef DEBUG
         printf("refl: %f\n", finish->refl);
#endif 
      } else if (!nextWord.compare("emissive")) {
         in >> finish->em;
#ifdef DEBUG
         printf("emissive: %d\n", finish->em);
#endif 
      } else if (!nextWord.compare("refraction")) {
         in >> finish->refr;
#ifdef DEBUG
         printf("refr: %f\n", finish->refr);
#endif 
      } else if (!nextWord.compare("ior")) {
         in >> finish->ior;
#ifdef DEBUG
         printf("ior: %f\n", finish->ior);
#endif 
      } else {
         cerr << "Bad formatting in finish"  << endl;
         return kBadFormat;
      }
      in >> nextWord;
   }

   //If there was more after the '}', put it back into the stream
   for (int i = nextWord.length() - 1; i > 0; i--) {
      in.unget();
   }

   return kSuccess;
}

int POVRayParser::parseScale(std::ifstream &in, glm::mat4 *matStack) {
   vec3 s;
   int status = parseVector(in, &s);
   
   if (status != kSuccess) return status;

   *matStack = glm::scale(mat4(1.0f), s) * (*matStack);

#ifdef DEBUG
   printf("scale: %f, %f, %f\n", s.x, s.y, s.z);
#endif 

   return kSuccess;
}

int POVRayParser::parseRotate(std::ifstream &in, glm::mat4 *matStack) {
   static const vec3 kXAxis = vec3(1.0f, 0.0f, 0.0f);
   static const vec3 kYAxis = vec3(0.0f, 1.0f, 0.0f);
   static const vec3 kZAxis = vec3(0.0f, 0.0f, 1.0f);

   vec3 r;
   int status = parseVector(in, &r);
   
   if (status != kSuccess) return status;

   // Euler rotations
   *matStack = glm::rotate(glm::mat4(1.0f), r.x, kXAxis) * (*matStack);
   *matStack = glm::rotate(glm::mat4(1.0f), r.y, kYAxis) * (*matStack);
   *matStack = glm::rotate(glm::mat4(1.0f), r.z, kZAxis) * (*matStack);

#ifdef DEBUG
   printf("rotate: %f, %f, %f\n", r.x, r.y, r.z);
#endif 

   return kSuccess;
}

int POVRayParser::parseTranslate(std::ifstream &in, glm::mat4 *matStack) {
   vec3 t;
   int status = parseVector(in, &t);
   
   if (status != kSuccess) return status;

   *matStack = glm::translate(mat4(1.0f), t) * (*matStack);

#ifdef DEBUG
   printf("translate: %f, %f, %f\n", t.x, t.y, t.z);
#endif 

   return kSuccess;
}

int POVRayParser::parseUV(std::ifstream &in, glm::vec2 *v) {
   char buffer[kBufferSize];
   int numArgs;

   in.getline(buffer, kBufferSize, '>');
   numArgs = sscanf(buffer, " < %f , %f > ", &v->x, &v->y);
   if (numArgs == 2)
      return kSuccess;
   else {
      cerr << "Bad format for uv entry" << endl;
      cerr << buffer << endl;
      return kBadFormat;
   }
}

int POVRayParser::parseVector(std::ifstream &in, glm::vec3 *v) {
   char buffer[kBufferSize];
   int numArgs;

   in.getline(buffer, kBufferSize, '>');
   numArgs = sscanf(buffer, " < %f , %f , %f > ", &v->x, &v->y, &v->z);
   if (numArgs == 3)
      return kSuccess;
   else {
      cerr << "Bad format for vector entry" << endl;
      cerr << buffer << endl;
      return kBadFormat;
   }
}

int POVRayParser::parseWord(std::ifstream &in, string s) {
   string nextWord;
   in >> nextWord;
   if (nextWord.compare(s)) {
      cerr << "Expected \"" << s << "\" while parsing, found: " << nextWord
           << endl;
      return kBadFormat;
   }
   return kSuccess;
}

int POVRayParser::parseCharacter(std::ifstream &in, char c) {
   string nextWord;

   in >> nextWord;
   if (nextWord[0] != c) {
      cerr << "Expected \"" << c << "\" while parsing, found: " << nextWord[0]
           << endl;
      return kBadFormat;
   }


   // Put everything but the first character back into the istream
   for (int i = nextWord.length() - 1; i > 0; i--) {
      in.unget();
      if (isspace(in.peek())) i++;
   }

   return kSuccess;
}

//TODO May not be safe if an error is caused with ifstream
bool POVRayParser::isComment(std::ifstream &in) {
   char c1, c2;
   string nextWord;
   bool isComment = false;
    
   if (in.eof()) return false;

   in >> nextWord;

   if (nextWord.length() >= 2) {
      isComment = nextWord[0] == '/' && nextWord[1] == '/';
   }

   // Put everything but the first character back into the istream
   for (int i = nextWord.length(); i > 0; i--) {
      in.unget();
      if (isspace(in.peek())) i++;
   }
   
   return isComment;
}

//TODO May not be safe if an error is caused with ifstream
bool POVRayParser::isUV(std::ifstream &in) {
   char c1, c2;
   string nextWord;
   bool isUV= false;
    
   if (in.eof()) return false;

   in >> nextWord;

   if (nextWord.length() >= 2) {
      isUV = nextWord[0] == 'u' && nextWord[1] == 'v';
   }

   // Put everything but the first character back into the istream
   for (int i = nextWord.length(); i > 0; i--) {
      in.unget();
      if (isspace(in.peek())) i++;
   }
   
   return isUV;
}

void POVRayParser::checkComments(std::ifstream &in) {
   char buffer[kBufferSize];
   while (isComment(in)) 
      in.getline(buffer, kBufferSize);
}
