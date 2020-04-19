#ifndef MATH_HH_INCLUDE
#define MATH_HH_INCLUDE

#include "Vector3.cuh"
#include "Mat4.cuh"

namespace math {
  
	template <typename T>
	__host__ __device__ inline T min(T const &a, T const &b) {
		return a < b ? a : b;
	}

	template <typename T>
	__host__ __device__ inline T max(T a, T b) {
		return a > b ? a : b;
	}
  __host__ __device__ inline mat4 translate_matrix(const Vector3 v) {
    
    return mat4(  1, 0, 0, v[0],
                  0, 1, 0, v[1],
                  0, 0, 1, v[2],
                  0, 0, 0, 1
    );
    
  }
  
  __host__ __device__ inline mat4 scale_matrix(const Vector3 v) {
    
    return mat4(  v[0],    0,    0, 0,
                     0, v[1],    0, 0,
                     0,    0, v[2], 0,
                     0,    0,    0, 1
    );
    
  }
  
  __host__ __device__ inline mat4 rotation_matrix(const float a, const Vector3 v) {
    
    float x = v[0];
    float y = v[1];
    float z = v[2];
    
    float ang = a * M_PI/180;
    
    float c = std::cos(ang);
    float s = std::sin(ang);
    float d = 1 - c;
    
    return mat4( (x*x*d) + c    , (x*y*d) - (z*s) , (x*z*d) + (y*s) , 0,
                 (y*x*d) + (z*s), (y*y*d) + c     , (y*z*d) - (x*s) , 0,
                 (z*x*d) - (y*s), (z*y*d) + (x*s) , (z*z*d) + c     , 0,
                        0       ,         0       ,         0       , 1
    );
    
  }

  __host__ __device__ inline mat4 translate(mat4 m, Vector3 v) {
    return m * translate_matrix(v);
  }
  
 __host__ __device__  inline mat4 scale(mat4 m, Vector3 v) {
    return m * scale_matrix(v);
  }
  
  __host__ __device__ inline mat4 scale(mat4 m, float s) {
    return m * scale_matrix(Vector3(s));
  }
  
  __host__ __device__ inline mat4 rotate(mat4 m, float ang, Vector3 v) {
    return m * rotation_matrix(ang, v);
  }

}

#endif /* MATH_HH_INCLUDE */
