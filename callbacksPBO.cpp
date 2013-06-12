//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm/glm.hpp"

using glm::vec3;

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;
extern void modifyCamera(vec3 trans, float rotHoriz, float rotVert); 
void setFilter(bool blur, bool median, bool motionBlur);

// The user must create the following routines:
void runCuda();
void saveImage(int frame);

void display()
{
   static int frame = 0;
   saveImage(frame++);
   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}

float kCamSpeed = 0.25f;
float kRotateSpeed = 3.0f;

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
   static bool blur = false, median = false, motionBlur = false;
   float tx = 0.0f, ty = 0.0f, tz = 0.0f; 
   float ry = 0.0f;
   float rz = 0.0f;
   bool cameraChange = false;

   switch(key) {
   case 27:
      exit(0);
      break;
   case 'q':
      tz = kCamSpeed;
      cameraChange = true;
      break;
   case 'e':
      tz = -kCamSpeed;
      cameraChange = true;
      break;
   case 's':
      ty = -kCamSpeed;
      cameraChange = true;
      break;
   case 'w':
      ty = kCamSpeed;
      cameraChange = true;
      break;
   case 'a':
      tx = -kCamSpeed;
      cameraChange = true;
      break;
   case 'd':
      tx = kCamSpeed;
      cameraChange = true;
      break;
   case 'i':
      rz = kRotateSpeed;
      cameraChange = true;
      break;
   case 'k':
      rz = -kRotateSpeed;
      cameraChange = true;
      break;
   case 'j':
      ry = kRotateSpeed;
      cameraChange = true;
      break;
   case 'l':
      ry = -kRotateSpeed;
      cameraChange = true;
      break;
   case 'b':
      blur = !blur;
      setFilter(blur, median, motionBlur);
      break;
   case 'm':
      median = !median;;
      setFilter(blur, median, motionBlur);
      break;
   case 'v':
      motionBlur = !motionBlur;;
      setFilter(blur, median, motionBlur);
      break;
   }
   if (cameraChange) modifyCamera(glm::vec3(tx, ty, tz), ry, rz);

   // indicate the display must be redrawn
   glutPostRedisplay();
}

// No mouse event handlers defined
void mouse(int button, int state, int x, int y)
{
}

void motion(int x, int y)
{
}
