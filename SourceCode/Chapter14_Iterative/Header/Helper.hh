#ifndef Helper_HH_INCLUDE
#define Helper_HH_INCLUDE
#include <iostream>
#include <stdint.h>

static const uint8_t clz_table_4bit[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };


class Helper {
  
public:

    template <typename T> static int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
    
    static unsigned int morton3D(float x, float y, float z)
    {
        x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
        y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
        z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
        unsigned int xx = expandBits((unsigned int)x);
        unsigned int yy = expandBits((unsigned int)y);
        unsigned int zz = expandBits((unsigned int)z);
        return xx * 4 + yy * 2 + zz;
    }
    
    static unsigned int clz32d( uint32_t x ) /* 32-bit clz */
    {
        unsigned int n = 0;
        if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
        if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
        if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
        n += (unsigned int)clz_table_4bit[x >> (32-4)];
        return n;
    }
    
private:
    
    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    static unsigned int expandBits(unsigned int v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }
    
};

#endif /* Helper_HH_INCLUDE */
