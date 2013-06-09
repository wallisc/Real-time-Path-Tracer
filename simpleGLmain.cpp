// simpleGLmain.cpp (Rob Farber)

// includes
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include "kernel.h"

using std::string;
const int kDefaultImageWidth = 640;
const int kDefaultImageHeight = 480;
const int kDefaultDepth = 4;

// The user must create the following routines:
// CUDA methods
extern void initCuda(string fileName, int depth, bool isStatic, char *outFile);
extern void runCuda();
extern void renderCuda(int);

// callbacks
extern void display();
extern void keyboard(unsigned char key, int x, int y);
extern void mouse(int button, int state, int x, int y);
extern void motion(int x, int y);

// GLUT specific variables
unsigned int window_width = kDefaultImageWidth;
unsigned int window_height = kDefaultImageHeight;


// Forward declarations of GL functionality
bool initGL(int argc, char** argv);
struct timeval start, end;

void printInputError() {
   printf("raytrace: must specify an input file\n");   
   printf("Try `raytrace --help` for more information\n");
}

int parseArgs(int argc, char *argv[], int *imgWidth, int *imgHeight, int *depth, 
              char **fileName, char **outFile, bool *isStatic) {
   bool imageWidthParsed = false;
   bool imageHeightParsed = false;
   *isStatic = false;

   if (argc < 2) { printInputError();
      return EXIT_FAILURE;
   }

   if (!strcmp(argv[1], "--help")) {
      printf("raytrace options are:\n");
      printf("\timageWidth\n\timageHeight\n\t-I sample.pov\n\t-O output.tga\n");
      printf("An example command to generate a 1024x800 image named \"image.tga\"" 
             " using \"input.pov\" with phong shading is :\n\n");
      printf("$raytrace 1024 800 -p -I input.pov -O image.tga\n");

      return EXIT_SUCCESS;
   }

   for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-' && argv[i][1] == 'O') {
         if (strlen(argv[i]) ==  2) {
            *outFile = argv[++i];
         } else {
            *outFile = argv[i] + 2;
         }
      } else if (argv[i][0] == '-' && argv[i][1] == 'd') {
            *depth = atoi(argv[++i]);
      } else if (argv[i][0] == '-' && argv[i][1] == 's') {
         *isStatic = true;
      } else if (argv[i][0] == '-' && argv[i][1] == 'I') {
         if (strlen(argv[i]) ==  2) {
            *fileName = argv[++i];
         } else {
            *fileName = argv[i] + 2;
         }
      } else {
         if (!imageWidthParsed) {
            *imgWidth= atoi(argv[i]);
            imageWidthParsed = true;
         } else if (!imageHeightParsed) {
            *imgHeight = atoi(argv[i]);
            imageHeightParsed = true;
         }
      }
   }

   if (!fileName) {
      printInputError();
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

// Simple method to display the Frames Per Second in the window title
void computeFPS()
{
   static int fpsCount=0;
   static int fpsLimit=100;

   fpsCount++;

   if (fpsCount == fpsLimit) {
      char fps[256];
      long seconds  = end.tv_sec  - start.tv_sec;
      long useconds = end.tv_usec - start.tv_usec;

      long utime = ((seconds) * 1000000 + useconds);
      double ifps = 1.0 /utime;//1.f / (mtime / 1000.f);
      ifps = ifps * (1000000);// 1/usec -> 1/sec
      sprintf(fps, "Path Tracer: %3.4f fps, Total Time: %3.4fms", ifps, (utime/(float)1000));

      glutSetWindowTitle(fps);
      fpsCount = 0;
   }
}

void fpsDisplay()
{
   gettimeofday( &start, NULL );
   display();

   gettimeofday( &end, NULL );
   computeFPS();
}

// Main program
int main(int argc, char** argv)
{
   int imgHeight = kDefaultImageHeight;
   int imgWidth = kDefaultImageWidth;
   char *fileName = NULL;
   char *outFile = "sample.tga";
   int status;
   int depth = kDefaultDepth;
   bool isStatic;
   
   status = parseArgs(argc, argv, &imgWidth, &imgHeight, &depth,
                      &fileName, &outFile, &isStatic);

   if (status == EXIT_FAILURE || !fileName)
      return status;


   window_width = imgWidth;
   window_height = imgHeight;
   if (false == initGL(argc, argv)) {
      return false;
   }

   initCuda(fileName, depth, isStatic, outFile);

   // register callbacks
   glutDisplayFunc(fpsDisplay);
   glutKeyboardFunc(keyboard);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);

   // start rendering mainloop
   glutMainLoop();
}

bool initGL(int argc, char **argv)
{
   //Steps 1-2: create a window and GL context (also register callbacks)
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(window_width, window_height);
   glutCreateWindow("Cuda GL Interop Demo (adapted from NVIDIA's simpleGL");
   glutDisplayFunc(fpsDisplay);
   glutKeyboardFunc(keyboard);
   glutMotionFunc(motion);

   // Step 3: Setup our viewport and viewing modes
   glViewport(0, 0, window_width, window_height);

   glClearColor(0.0, 0.0, 0.0, 1.0);
   glDisable(GL_DEPTH_TEST);


   // set view matrix
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

   return true;
}
