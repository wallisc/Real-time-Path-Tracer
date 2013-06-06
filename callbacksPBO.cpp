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
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();
extern void modifyCamera(vec3 trans, float rotHoriz, float rotVert); 

// The user must create the following routines:
void runCuda();
void saveImage();

void display()
{
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
   float tx = 0.0f, ty = 0.0f, tz = 0.0f; 
   float ry = 0.0f;
   float rz = 0.0f;
   switch(key) {
   case 'q' :
      saveImage();
      exit(0);
      break;
   case 'w':
      ty = -kCamSpeed;
      break;
   case 's':
      ty = kCamSpeed;
      break;
   case 'a':
      tx = kCamSpeed;
      break;
   case 'd':
      tx = -kCamSpeed;
      break;
   case 'i':
      rz = kRotateSpeed;
      break;
   case 'k':
      rz = -kRotateSpeed;
      break;
   case 'j':
      ry = kRotateSpeed;
      break;
   case 'l':
      ry = -kRotateSpeed;
      break;
   }
   modifyCamera(vec3(tx, ty, tz), ry, rz); 

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
