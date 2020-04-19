#ifndef _PDF_SHAPE_HH_INCLUDE
#define _PDF_SHAPE_HH_INCLUDE

#include "Triangle.cuh"

class Shape_list {

public:
  
  __host__ __device__ Shape_list() {};
  __host__ __device__ Shape_list(Triangle* p, int n) list(p), size(n) {};
  
  float pdf_value(const Vector3 &origin, cont Vector3 &direction){
    float weight = 1.0/size;
    float sum = 0;
    
    for(int i = 0; i < list; i++)
      sum += weight*list[i]->pdf_value(origin, direction);
    
    return sum;
    
  }
  
private:
  
  Triangle *list;
  int size;
  
};

class pdf{
  
public:

  __host__ __device__ pdf() {};
  __host__ __device__ pdf(Shape_list* p, Vector3 &o) ptr(p), origin(o) {};
  
  float value(const Vector3& direction){
    return ptr->pdf_value(origin, direction);
  }
  
private:
  
  Vector3 origin;
  Shape_list *ptr;
  
};

class mixture_pdf()

#endif /* _SKYBOX_HH_INCLUDE */
