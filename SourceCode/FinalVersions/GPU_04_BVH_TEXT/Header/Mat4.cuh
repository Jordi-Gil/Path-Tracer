#ifndef MAT4_HH_INCLUDE
#define MAT4_HH_INCLUDE

#include <ostream>

class mat4 {

public:

  __host__ __device__ mat4() {}
  __host__ __device__ mat4(float val);
  __host__ __device__ mat4(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4, 
    float c1, float c2, float c3, float c4, float d1, float d2, float d3, float d4
  );
  __host__ __device__ mat4(float m[4][4]);

  __host__ __device__ float operator()(int i, int j) const;
  __host__ __device__ float& operator()(int i, int j);
  
  __host__ __device__ mat4 operator*(const mat4 &v2);

  __host__ __device__ static mat4 identity();
    
  float matrix[4][4];

};

__host__ inline std::ostream& operator<<(std::ostream &os, const mat4 &t){
    os << t.matrix[0][0] << " " << t.matrix[0][1] << " " << t.matrix[0][2] << " " << t.matrix[0][3] << "\n";
    os << t.matrix[1][0] << " " << t.matrix[1][1] << " " << t.matrix[1][2] << " " << t.matrix[1][3] << "\n";
    os << t.matrix[2][0] << " " << t.matrix[2][1] << " " << t.matrix[2][2] << " " << t.matrix[2][3] << "\n";
    os << t.matrix[3][0] << " " << t.matrix[3][1] << " " << t.matrix[3][2] << " " << t.matrix[3][3] << "\n";
    return os;
}

#endif /* MAT4_HH_INCLUDE */
