#ifndef Helper_HH_INCLUDE
#define Helper_HH_INCLUDE
#include <iostream>
#include <stdint.h>

#include "Math.cuh"

static const uint8_t clz_table_4bit[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

class Helper {
  
public:
    
  __host__ __device__ static int sgn(int val) {
    return (int(0) < val) - (val < int(0));
  }
    
  __host__ __device__ static unsigned long long int clz32d( uint32_t x ) /* 32-bit clz */ {
      unsigned  long long int  n = 0;
      if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
      if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
      if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
      n += (unsigned  long long int )clz_table_4bit[x >> (32-4)];
      return n;
  }
    
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
  __host__ __device__ static unsigned  long long int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
  }
    
  __host__ __device__ static unsigned long long int morton3D(float x, float y, float z) {
    x = math::min(math::max(x * 1024.0f, 0.0f), 1023.0f);
    y = math::min(math::max(y * 1024.0f, 0.0f), 1023.0f);
    z = math::min(math::max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned  long long int xx = expandBits((unsigned int)x);
    unsigned  long long int yy = expandBits((unsigned int)y);
    unsigned  long long int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
  }
};

#endif /* Helper_HH_INCLUDE */
