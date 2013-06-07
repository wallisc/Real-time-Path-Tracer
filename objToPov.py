# Converts a .obj file to a povray file using smooth_triangles
# Christopher Wallis
# Example usage: 'python objToPov.py bunny.obj'

import sys, re 

DEFAULT_CAMERA = """
camera {
    location <0, 0, 1>
    up <0, 1, 0>
    right <1.33, 0, 0>
    look_at <0, 0, 0>
}
"""

DEFAULT_LIGHTS = """
light_source { <0, 1.5 , 0> color rgb <1.0, 1.0, 1.0> }
"""

class Material:
   clr = (0.0, 0.0, 0.0)
   alpha = 0.0
   amb = 0.0
   dif = 1.0
   spec = 0.0
   rough = 0.0
   refl = 0.0
   refr = 0.0
   ior = 1.0

defMat = Material()
defMat.clr = (.5, .5, .5)
defMat.alpha = 0.0
defMat.amb = 0.2
defMat.dif = 0.8
defMat.spec = 0.3
defMat.rough = 0.1
defMat.refl = 0.0
defMat.refr = 0.0
defMat.ior = 1.0

def addMaterials(fileName, matMap):
   file = open(fileName, 'r')

   newMat = re.compile(r"\s*newmtl\s.*")
   illum = re.compile(r"\s*illum\s.*")
   specCoef = re.compile(r"\s*Ns\s.*")
   alpha = re.compile(r"\s*tr\s.*")
   diffuse = re.compile(r"\s*d\s.*")
   pigment = re.compile(r"\s*Kd\s.*")
   ior = re.compile(r"\s*Ni\s.*")

   line = file.readline()
   while line != '':
      if newMat.match(line):
         tokens = re.split("\s+", line)
         matName = tokens[1]
         mat = Material()
         line = file.readline()
         while not newMat.match(line):
            if specCoef.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               mat.rough = 1.0 / float(tokens[idx]) 
            elif alpha.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               mat.alpha= float(tokens[idx])
            elif diffuse.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               mat.dif = float(tokens[idx])
            elif pigment.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               mat.clr = (float(tokens[idx]), float(tokens[idx + 1]), float(tokens[idx + 2]))
            elif ior.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               mat.ior = float(tokens[idx])
            elif illum.match(line):
               tokens = re.split("\s+", line)
               idx = 2 if tokens[0] == "" else 1
               illumNum = int(tokens[idx])
               if illumNum == 5:
                  mat.refl = 1.0
                  mat.dif = 0.0
               elif illumNum == 6 or illumNum == 7:
                  mat.refr = 1.0
                  mat.alpha = 1.0
                  mat.dif = 0.0
               
            elif line == '':
               matMap[matName] = mat 
               #for mat, m in matMap.iteritems():
               #   print mat + ": " + " color is " + repr(m.clr[0]) + ", " + repr(m.clr[1]) + ", " + repr(m.clr[2]) 
               return

            line = file.readline()
         matMap[matName] = mat 
         continue
      else:
         line = file.readline()

def matToStr(mat):
   if mat.alpha > 0.0:
      str = "   pigment { color rgbf < " + repr(mat.clr[0]) + " , " + repr(mat.clr[1]) + " , " + repr(mat.clr[2]) + " , " + repr(mat.alpha) + " }\n"
   else:
      str = "   pigment { color rgb < " + repr(mat.clr[0]) + " , " + repr(mat.clr[1]) + " , " + repr(mat.clr[2]) + " }\n"
   str += "   finish { ambient " + repr(mat.amb) + " diffuse " + repr(mat.dif) + " specular " + repr(mat.spec) + " roughness " + repr(mat.rough)   

   if mat.refl > 0.0:
      str += " reflection " + repr(mat.refl)

   if mat.refr > 0.0:
      str += " refraction " + repr(mat.refr) + " ior " + repr(mat.ior)

   str += " }\n"
   return str


def vecToStr(vec):
   return "<" + ", ".join(["%.6f" % i for i in vec]) + ">"

def convertToPOV(fileName):
   objCheck = re.compile(r".*\.obj")
            
   if objCheck.match(fileName) == None:
      print fileName + " is not a valid .obj file\n"
      return
      
   matMap = {}

   file = open(fileName, 'r')
   outputName = fileName[:len(fileName) - 4] + ".pov"
   outFile = open(outputName, 'w')

   outFile.write(DEFAULT_CAMERA)
   outFile.write(DEFAULT_LIGHTS)

   normList = []
   vertList = []
   texList = []
   normalCheck = re.compile(r"vn\s.*")
   matLibCheck = re.compile(r"mtllib\s.*")
   useMat = re.compile(r"usemtl\s.*")
   vertexCheck = re.compile(r"\s*v\s.*")
   textureCheck = re.compile(r"\s*vt\s.*")
   faceCheck = re.compile(r"\s*f\s.*")
   whiteSpaceCheck = re.compile(r"\s*")
   
   line = file.readline()

   faceCount = 0

   curMat = defMat
   while line != '':
      if matLibCheck.match(line):
         files = re.split("\s*", line)
         addMaterials(files[1], matMap)

      elif useMat.match(line):
         tokens = re.split("\s*", line)
         matName = tokens[1]
         curMat = matMap[matName]

      elif normalCheck.match(line):
         normals = re.split("\s*", line)
         # Skip the vn tag (i.e. start at index 1)
         n1 = float(normals[1])
         n2 = float(normals[2])
         n3 = float(normals[3])
         normList.append((n1, n2, n3))

      elif vertexCheck.match(line):
         vertex = re.split("\s*", line)
         # Skip the v tag (i.e. start at index 1)
         v1 = float(vertex[1])
         v2 = float(vertex[2])
         v3 = float(vertex[3])
         vertList.append((v1, v2, v3))

      elif textureCheck.match(line):
         vertex = re.split("\s*", line)
         # Skip the vt tag (i.e. start at index 1)
         v1 = float(vertex[1])
         v2 = float(vertex[2])
         texList.append((v1, v2))
      
      elif faceCheck.match(line):
         objFaces = 0
         while faceCheck.match(line): 

            if faceCheck.match(line):
               faceCount += 1
               face = re.split("\s+", line)
               uvIdxList = []

               # Handle if there are 4 verticies supplied
               if (len(face) == 5 and not whiteSpaceCheck.match(face[4])) or (len(face) == 6 and face[5] == ""):
                  print line
                  faceCount += 1
                  point1 = re.split("/", face[1])
                  point2 = re.split("/", face[2])
                  point3 = re.split("/", face[3])
                  point4 = re.split("/", face[4])
                  v1idx = int(point1[0]) - 1
                  v2idx = int(point2[0]) - 1
                  v3idx = int(point3[0]) - 1
                  v4idx = int(point4[0]) - 1

                  # If using relative indicies
                  if v1idx < 0:
                     v1idx = len(vertList) + v1idx + 1
                     v2idx = len(vertList) + v2idx + 1
                     v3idx = len(vertList) + v3idx + 1
                     v4idx = len(vertList) + v4idx + 1

                  outFile.write("triangle {\n")
                  outFile.write("   " + vecToStr(vertList[v1idx]) + ", ")
                  outFile.write(vecToStr(vertList[v2idx]) + ", ")
                  outFile.write(vecToStr(vertList[v3idx]) + "\n") 
                  outFile.write(matToStr(curMat))
                  outFile.write("}\n\n")

                  outFile.write("triangle {\n")
                  outFile.write("   " + vecToStr(vertList[v1idx]) + ", ")
                  outFile.write(vecToStr(vertList[v3idx]) + ", ")
                  outFile.write(vecToStr(vertList[v4idx]) + "\n") 
                  outFile.write(matToStr(curMat))
                  outFile.write("}\n\n")


               #if no normals were specified
               if len(re.split("/", face[1])) <= 2:
                  outFile.write("triangle {\n")
                  point1 = re.split("/", face[1])
                  point2 = re.split("/", face[2])
                  point3 = re.split("/", face[3])
                  v1idx = int(point1[0]) - 1
                  v2idx = int(point2[0]) - 1
                  v3idx = int(point3[0]) - 1

                  # If using relative indicies
                  if v1idx < 0:
                     v1idx = len(vertList) + v1idx + 1
                     v2idx = len(vertList) + v2idx + 1
                     v3idx = len(vertList) + v3idx + 1

                  outFile.write("   " + vecToStr(vertList[v1idx]) + ", ")
                  outFile.write(vecToStr(vertList[v2idx]) + ", ")
                  outFile.write(vecToStr(vertList[v3idx]) + "\n") 

               else:
                  outFile.write("smooth_triangle {\n")
                  for idx in range(1, 4):
                     point = re.split("/", face[idx])
                     # Note: OBJ files are 1-indexed
                     vertIdx = int(point[0]) - 1 
                     
                     
                     # If vertex textures were specified
                     if point[1] != "":
                        uvIdxList.append(int(point[1]) - 1)
                     
                     normIdx = int(point[len(point) - 1]) - 1
                     outFile.write("   " + vecToStr(vertList[vertIdx]) + ", " 
                                         + vecToStr(normList[normIdx]))
                     
                     if idx < 3: outFile.write(",\n")
                     else: outFile.write("\n")

               #if len(uvIdxList) == 3:
                  #outFile.write("   uv {")
                  #for idx in range(0, 3):
                  #   outFile.write(vecToStr(texList[uvIdxList[idx]]))
                  #   if idx < 2: outFile.write(", ")
                  #   else: outFile.write("}\n")


               outFile.write(matToStr(curMat))
               outFile.write("}\n\n")

            line = file.readline()
         continue

      line = file.readline()

         

   print "Succesfully wrote out " + outputName
   print "Number of triangles: " + repr(faceCount)

def main():
   args = sys.argv
   for i in range(1, len(args)):
      convertToPOV(args[i])

main()
