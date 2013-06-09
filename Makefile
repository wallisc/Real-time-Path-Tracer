CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -g
LDFLAGS= -O3 -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU 
CUDAFLAGS= -O2 -lineinfo -g -c -arch=sm_21 -Xptxas -dlcm=ca -prec-div=false -prec-sqrt=false -use_fast_math

ALL= callbacksPBO.o simpleGLmain.o simplePBO.o kernel.o POVRayParser.o Image.o

HEADERS = cudaError.h kernel.h Plane.h Shader.h Geometry.h Light.h PointLight.h Ray.h 
HEADERS += TokenData.h Material.h Util.h PhongShader.h CookTorranceShader.h 
HEADERS += Triangle.h bvh.h GeometryUtil.h BoundingBox.h bvh.cpp

all= $(ALL)  RTRT

RT: $(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o raytrace 

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch junk* raytrace

POVRayParser.o: POVRayParser.cpp POVRayParser.h
	$(CC) $(CFLAGS) -c POVRayParser.cpp

kernel.o: kernel.cu $(HEADERS)
	$(CC) $(CUDAFLAGS) -c  kernel.cu

Image.o: Image.cpp Image.h
	$(CC) $(CFLAGS) -c Image.cpp
