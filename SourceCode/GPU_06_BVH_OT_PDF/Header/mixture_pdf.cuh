#ifndef _MIXTURE_PDF_HH_INCLUDE
#define _MIXTURE_PDF_HH_INCLUDE

#include <curand.h>
#include <curand_kernel.h>

#include "pdf.cuh"

class mixture_pdf {
  
  
public:

  __host__ __device__ mixture_pdf() {};
  __host__ __device__ mixture_pdf(pdf pdf0, pdf pdf1);
  
  __device__ float value(const Vector3 &direction);
  
  __device__ Vector3 generate(curandState *_random);
  
private:
  
  pdf pdfs[2];
  
};


#endif /* _MIXTURE_PDF_HH_INCLUDE */
