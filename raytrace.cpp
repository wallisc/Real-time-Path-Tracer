#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <string.h> 

#include "kernel.h"
#include "POVRayParser.h"
#include "Image.h"

const int kDefaultImageWidth = 640;
const int kDefaultImageHeight = 480;
const int kDefaultPasses = 2;
const int kDefaultDepth = 2;

void printInputError(); 
int parseArgs(int argc, char *argv[], int *imgWidth, int *imgHeight, int *depth, int *passes,
              char **fileName, char **outFile, ShadingType *stype);

int main(int argc, char *argv[]) {
   int imgHeight = kDefaultImageHeight;
   int imgWidth = kDefaultImageWidth;
   char *fileName = NULL;
   char *outFile = "sample.tga";
   ShadingType stype = PHONG;
   int status;
   int passes = kDefaultPasses;
   int depth = kDefaultDepth;

   status = parseArgs(argc, argv, &imgWidth, &imgHeight, &depth, &passes, 
                      &fileName, &outFile, &stype);
   // If the arguments were incorrect or the user only used the --help flag
   if (status == EXIT_FAILURE || !fileName)
      return status;
  
   TKSceneData data;
   status = POVRayParser::parseFile(fileName, &data);
   if (status != POVRayParser::kSuccess) {
      printf("Error parsing file\n");
      return EXIT_FAILURE;
   }

   // Do the actual ray tracing
   launch_kernel(&data, stype, imgWidth, imgHeight, depth, passes, output);


}

int parseArgs(int argc, char *argv[], int *imgWidth, int *imgHeight, int *depth, int *passes,
              char **fileName, char **outFile, ShadingType *stype) {

   bool imageWidthParsed = false;
   bool imageHeightParsed = false;

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
      } else if (argv[i][0] == '-' && argv[i][1] == 'n') {
            *passes = atoi(argv[++i]);
      } else if (argv[i][0] == '-' && argv[i][1] == 'I') {
         if (strlen(argv[i]) ==  2) {
            *fileName = argv[++i];
         } else {
            *fileName = argv[i] + 2;
         }
      } else if (argv[i][0] == '-' && argv[i][1] == 'p') {
        *stype = PHONG; 
      } else if (argv[i][0] == '-' && argv[i][1] == 't') {
        *stype = COOK_TORRANCE; 
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

