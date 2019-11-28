#ifndef Shape_HH_INCLUDE
#define Shape_HH_INCLUDE

#include "Sphere.hh"
#include "Triangle.hh"

enum ShapeType {SPHERE, TRIANGLE};

struct Shape {
  
  ShapeType tag;
  union {
    Sphere s;
    Triangle t;
  };
  
  Shape() {}
  
};

struct ObjEval{
      
  inline bool operator()(Triangle a, Triangle b){
  
    return a.getMorton() < b.getMorton();
  }
};

struct ObjEval2{
      
  inline bool operator()(Sphere a, Sphere b){
  
    return a.getMorton() < b.getMorton();
  }
};

/*

struct ObjEval{
  
  inline bool compare(Sphere a, Sphere b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Triangle a, Triangle b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Triangle a, Sphere b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Sphere a, Triangle b) { return a.getMorton() < b.getMorton(); }
  
    
  inline bool operator()(Shape a, Shape b){
  
    if(a.tag == SPHERE) {
      if(b.tag == SPHERE) return compare(a.s, b.s);
      else return compare(a.s, b.t);
    }
    else{
      if(b.tag == SPHERE) return compare(a.t, b.s);
      else return compare(a.t, b.t);
    }
    
  }

};

*/


#endif /* Shape_HH_INCLUDE */
