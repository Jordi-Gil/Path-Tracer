#ifndef _VECTOR3_INCLUDE
#define _GAME_INCLUDE

class Vector3
{
    
public:
    
    Vector3(int x, int y, int z);
    
    static Vector3 Zero(void);
    static Vector3 One(void);
    
private:
    int x;
    int y;
    int z;
    
    
}
