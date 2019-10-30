#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <limits>
#include <algorithm>
#include <stack>
#include <queue>

#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "MovingSphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Node.cuh"

#define MAX 3.402823466e+38
#define MIN 1.175494351e-38
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define cuRandom (curand_uniform(&local_random))
#define Random (rand()/(RAND_MAX + 1.0))

void print(MovingSphere *h_objects,int size){
  for(int i = 0; i < size; i++){
    std::cout << h_objects[i].center << " " << h_objects[i].mat_ptr->name << " " << h_objects[i].mat_ptr->albedo << " " << h_objects[i].mat_ptr->getAlbedo() << std::endl;
  }
  
  MovingSphere a = MovingSphere(Vector3(1,1,1),Vector3(1,1,1),0.0,1.0,0.2, new Dielectric(1.5));
  std::cout << a.center << " " << a.mat_ptr->name << " " << a.mat_ptr->albedo << " " << a.mat_ptr->getAlbedo() << std::endl;
  
  Material *mat = new Lambertian(Vector3(2,2,2));
  std::cout << "Lambertian? " << mat->name << " " << mat->albedo << std::endl;
  
}

void compare(Vector3 &max, Vector3 &min, Vector3 point) {
    
  if(point[0] > max[0]) max[0] = point[0]; //x
  if(point[1] > max[1]) max[1] = point[1]; //y
  if(point[2] > max[2]) max[2] = point[2]; //z
  
  if(point[0] < min[0]) min[0] = point[0]; //x
  if(point[1] < min[1]) min[1] = point[1]; //y
  if(point[2] < min[2]) min[2] = point[2]; //z
}


void create_world(MovingSphere *h_objects, int &size, int dist){

  Vector3 max(MIN);
  Vector3 min(MAX);
  
  int i = 0;
  //i++;
  for (int a = -dist; a < dist; a++) {
    for (int b = -dist; b < dist; b++) {
      float material = Random;
      Vector3 center(a+0.9*Random, 0.2, b+0.9*Random);
      
      //if ((center-Vector3(0,0,0)).length() > 0.995) {
        if (material < 0.8)
          h_objects[i] = MovingSphere(center, center+Vector3(0,0.5*Random,0), 0.0, 1.0, 0.2, new Lambertian(Vector3(Random*Random, Random*Random, Random*Random)));
        else if (material < 0.95) 
          h_objects[i] = MovingSphere(center, center, 0.0, 1.0, 0.2, new Metal(Vector3(0.5*(1.0+Random), 0.5*(1.0+Random), 0.5*(1.0+Random)),0.5*Random));
        else 
          h_objects[i] = MovingSphere(center, center, 0.0, 1.0, 0.2, new Dielectric(1.5));
        
        compare(max,min,h_objects[i].getCenter());
        i++;
          
      //}
    }
  }
    
  std::cout << i << std::endl;
  
  h_objects[0] = MovingSphere(Vector3(0,-1000,-1), Vector3(0,-1000,-1), 0.0, 1.0, 1000, new Lambertian(Vector3(0.5, 0.5, 0.5))); 
  compare(max,min, h_objects[0].getCenter());
  
  h_objects[1] = MovingSphere(Vector3( 0, 1, 0), Vector3( 0, 1, 0), 0.0, 1.0, 1.0, new Dielectric(1.5));
  compare(max,min,h_objects[1].getCenter()); //i++;
    
  h_objects[2] = MovingSphere(Vector3(-4, 1, 0), Vector3(-4, 1, 0), 0.0, 1.0, 1.0, new Lambertian(Vector3(0.4, 0.2, 0.1)));
  compare(max,min,h_objects[2].getCenter()); //i++;
  
  h_objects[3] = MovingSphere(Vector3( 4, 1, 0), Vector3( 4, 1, 0), 0.0, 1.0, 1.0, new Metal(Vector3(0.7, 0.6, 0.5),0.0));
  compare(max,min,h_objects[3].getCenter()); //i++;
  
  h_objects[4] = MovingSphere(Vector3( 4, 1, 5), Vector3( 4, 1, 5), 0.0, 1.0, 1.0, new Metal(Vector3(0.9, 0.2, 0.2),0.0));
  compare(max,min,h_objects[4].getCenter()); //i++;
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
  size = i;
    
  for(int idx = 0; idx < size; idx++){
    Vector3 point = h_objects[idx].getCenter();
          
    point[0] = ((point[0] - min[0])/(max[0] - min[0]));
    point[1] = ((point[1] - min[1])/(max[1] - min[1]));
    point[2] = ((point[2] - min[2])/(max[2] - min[2]));
      
    h_objects[idx].setMorton(Helper::morton3D(point[0],point[1],point[2])+idx);
  }
    
  std::sort(h_objects, h_objects + size , ObjEval());
}

int main(){
  
  
  MovingSphere h_objects[16];
  //cudaMallocHost((MovingSphere **)&h_objects,16*sizeof(MovingSphere));
  int size;
  create_world(h_objects,size,2);
  
  std::cout << size << std::endl;
  print(h_objects,size);
  
}
