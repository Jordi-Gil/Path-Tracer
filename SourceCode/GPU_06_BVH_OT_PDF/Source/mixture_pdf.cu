#include "mixture_pdf.cuh"

__host__ __device__ mixture_pdf::mixture_pdf(pdf pdf0, pdf pdf1) {
  pdfs[0] = pdf0;
  pdfs[1] = pdf1;
}
  
__device__ float mixture_pdf::value(const Vector3 &direction){
  
  return 0.5 * pdfs[0].value(direction) + 0.5 * pdfs[1].value(direction);
  
}

__device__ Vector3 mixture_pdf::generate(curandState *_random){
  
  if(curand_uniform(_random) < 0.5){
    return pdfs[0].generate(_random);
  }
  else{
    return pdfs[1].generate(_random);
  }
  
}
  
