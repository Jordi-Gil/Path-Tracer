#include "Vector3.h"

Vector3(int x, int y, int z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

static Vector3 Zero(void)
{
    return Vector3(0,0,0);
}


static Vector3 One(void)
{
    return Vector3(1,1,1);
}
