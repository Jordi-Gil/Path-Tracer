#include "Skybox.cuh"

#define STB_IMAGE_STATIC
#include "stb_image.h"

void Skybox::load(const std::string &dir){
  
  int nx, ny, nn;
  unsigned char *image;
  
  std::string filename;
  
  //Front
  filename = "Textures/" + dir + "/front.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[0] = Rectangle(bottomLeft.x(), topRight.x(), bottomLeft.y(), topRight.y(), topRight.z(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, true, true, true)), FRONT);
  
  //Back
  filename = "Textures/" + dir + "/back.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[1] = Rectangle(bottomLeft.x(), topRight.x(), bottomLeft.y(), topRight.y(), bottomLeft.z(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, false, true, true)), BACK);
  
  //Top
  filename = "Textures/" + dir + "/top.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[2] = Rectangle(bottomLeft.x(), topRight.x(), bottomLeft.z(), topRight.z(), topRight.y(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, true, false, true)), TOP);
  
  //Bottom
  filename = "Textures/" + dir + "/bottom.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[3] = Rectangle(bottomLeft.x(), topRight.x(), bottomLeft.z(), topRight.z(), bottomLeft.y(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, true, true, true)), BOTTOM);
  
  //Left
  filename = "Textures/" + dir + "/left.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[4] = Rectangle(bottomLeft.y(), topRight.y(), bottomLeft.z(), topRight.z(), topRight.x(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, true, false, false)), LEFT);
  
  //Right
  filename = "Textures/" + dir + "/right.jpg";
  image = stbi_load(filename.c_str(), &nx, &ny, &nn, 0);
  list[5] = Rectangle(bottomLeft.y(), topRight.y(), bottomLeft.z(), topRight.z(), bottomLeft.x(), Material(SKYBOX, Texture(IMAGE, Vector3::Zero(), image, nx, ny, true, true, false)), RIGHT);
  
}

Skybox::Skybox(Vector3 a, Vector3 b, const std::string &dir) {
  
  bottomLeft = a;
  topRight = b;
  
  load(dir);
  
}

__device__ bool Skybox::hit(const Ray& r, float t_min, float t_max, hit_record& rec){
  
  hit_record temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_max;
  for(int i = 0; i < 6; i++){
    if(list[i].hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  
  return hit_anything;
  
}
