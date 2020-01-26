#include "Mat4.cuh"

__host__ __device__ mat4::mat4(float val){
  for (int i = 0; i < 4; i++){
    matrix[i][0] = val;
    matrix[i][1] = val;
    matrix[i][2] = val;
    matrix[i][3] = val;
  }
}

__host__ __device__ mat4::mat4(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4, float c1, float c2, float c3, float c4, float d1, float d2, float d3, float d4) {
  for(int i = 0; i < 4; i++) {
    if(i == 0) {
      matrix[i][0] = a1;
      matrix[i][1] = a2;
      matrix[i][2] = a3;
      matrix[i][3] = a4;
    }
    else if (i == 1) {
      matrix[i][0] = b1;
      matrix[i][1] = b2;
      matrix[i][2] = b3;
      matrix[i][3] = b4;
    }
    else if (i == 2) {
      matrix[i][0] = c1;
      matrix[i][1] = c2;
      matrix[i][2] = c3;
      matrix[i][3] = c4;
    }
    else {
      matrix[i][0] = d1;
      matrix[i][1] = d2;
      matrix[i][2] = d3;
      matrix[i][3] = d4;
    }
  }
}

__host__ __device__ float mat4::operator()(int i, int j) const{
  return matrix[i][j];
};

__host__ __device__ float& mat4::operator()(int i, int j) {
  return matrix[i][j];
};

__host__ __device__ mat4 mat4::operator*(const mat4 &m2){
  
  mat4 res;
  
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      
      res(i,j) = 0;
      for(int k = 0; k < 4; k++) {
        res(i,j) += matrix[i][k] * m2(k,j);
      }
    }
  }
  return res;
}

__host__ __device__ mat4 mat4::identity(){
	return mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
}
