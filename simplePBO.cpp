// simplePBO.cpp (Rob Farber)

// includes
#include <GL/glut.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <string>

#include "glm/glm.hpp"
#include "kernel.h"
#include "Image.h"
#include "POVRayParser.h"

using std::string;
using glm::vec3;

// external variables
extern float animTime;
extern unsigned int window_width;
extern unsigned int window_height;

// constants (the following should be a const in a header file)
unsigned int image_width = window_width;
unsigned int image_height = window_height;

// variables
GLuint pbo=0;
GLuint textureID=0;

void createPBO(GLuint* pbo)
{

   if (pbo) {
      // set up vertex data parameter
      int num_texels = image_width * image_height;
      int num_values = num_texels * 4;
      int size_tex_data = sizeof(GLubyte) * num_values;

      // Generate a buffer ID called a PBO (Pixel Buffer Object)
      glGenBuffers(1,pbo);
      // Make this the current UNPACK buffer (OpenGL is state-based)
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
      // Allocate data for the buffer. 4-channel 8-bit image
      glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
      cudaGLRegisterBufferObject( *pbo );
   }
}

void deletePBO(GLuint* pbo)
{
   if (pbo) {
      // unregister this buffer object with CUDA
      cudaGLUnregisterBufferObject(*pbo);

      glBindBuffer(GL_ARRAY_BUFFER, *pbo);
      glDeleteBuffers(1, pbo);

      *pbo = 0;
   }
}

void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
   // Enable Texturing
   glEnable(GL_TEXTURE_2D);

   // Generate a texture identifier
   glGenTextures(1,textureID);

   // Make this the current texture (remember that GL is state-based)
   glBindTexture( GL_TEXTURE_2D, *textureID);

   // Allocate the texture memory. The last parameter is NULL since we only
   // want to allocate memory, not initialize it
   glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, image_width, image_height, 0,
         GL_BGRA,GL_UNSIGNED_BYTE, NULL);

   // Must set the filter mode, GL_LINEAR enables interpolation when scaling
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
   // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
   // GL_TEXTURE_2D for improved performance if linear interpolation is
   // not desired. Replace GL_LINEAR with GL_NEAREST in the
   // glTexParameteri() call

}

void deleteTexture(GLuint* tex)
{
   glDeleteTextures(1, tex);

   *tex = 0;
}

void cleanupCuda()
{
   if(pbo) deletePBO(&pbo);
   if(textureID) deleteTexture(&textureID);
}

int g_depth;
int g_pass = 1;

// Run the Cuda part of the computation
void runCuda()
{
   uchar4 *dptr=NULL;

   // map OpenGL buffer object for writing from CUDA on a single GPU
   // no data is moved (Win & Linux). When mapped to CUDA, OpenGL
   // should not use this buffer
   cudaGLMapBufferObject((void**)&dptr, pbo);

   // execute the kernel
   launch_kernel(image_width, image_height, g_depth, g_pass++, dptr);

   // unmap buffer object
   cudaGLUnmapBufferObject(pbo);
}

void getImage(uchar4 *image) {
   uchar4 *dptr=NULL;

   cudaGLMapBufferObject((void**)&dptr, pbo);
   kernelGetImage(dptr, image, image_width, image_height);

   cudaGLUnmapBufferObject(pbo);
}

char *outFile;

void initCuda(string fileName, int depth, ShadingType stype, char *out)
{
   // First initialize OpenGL context, so we can properly set the GL
   // for CUDA.  NVIDIA notes this is necessary in order to achieve
   // optimal performance with OpenGL/CUDA interop.  use command-line
   // specified CUDA device, otherwise use device with highest Gflops/s
   int devCount= 0;
   g_depth = depth;
   cudaGetDeviceCount(&devCount);
   outFile = out;
   if( devCount < 1 )
   {
      printf("No GPUS detected\n");
      exit(EXIT_FAILURE);
   }
   cudaGLSetGLDevice( 0 );

   createPBO(&pbo);
   createTexture(&textureID,image_width,image_height);

   TKSceneData data;
   int status = POVRayParser::parseFile(fileName, &data);
   if (status != POVRayParser::kSuccess) {
      printf("Error parsing file\n");
      exit(EXIT_FAILURE);
   }

   // Clean up on program exit
   atexit(cleanupCuda);
   init_kernel(data, stype, image_width, image_height);
   runCuda();
}

void modifyCamera(vec3 translate, float rotHoriz, float rotVertical) {
   translateCamera(translate);
   rotateCameraVertically(rotVertical);
   rotateCameraSideways(rotHoriz);
   g_pass = 1;
}

void saveImage() {
   uchar4 *output = (uchar4 *)malloc(image_width * image_height * sizeof(uchar4));
   getImage(output);
   Image img(image_width, image_height);
   for (int x = 0; x < image_width; x++) {
      for (int y = 0; y < image_height; y++) {
         color_t clr;
         int idx = y * image_width + x;
         clr.r = output[idx].x / 255.0; clr.g = output[idx].y / 255.0; 
         clr.b = output[idx].z / 255.0; clr.f = 1.0;
         // Flip the image
         img.pixel(x, image_height - y - 1, clr);
      }
   }
   img.WriteTga(outFile);
}
