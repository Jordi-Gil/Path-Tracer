#include "Vector3.hh"

Vector3 unit_vector(Vector3 v){
    return v / v.length();
}
