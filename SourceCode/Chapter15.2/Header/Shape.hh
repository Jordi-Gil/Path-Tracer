#ifndef Shape_HH_INCLUDE
#define Shape_HH_INCLUDE

#include "Sphere.hh"
#include "Triangle.hh"

enum ShapeType {SPHERE, TRIANGLE};

struct Shape {
  
  Shape() : s() {}
  Shape(const Shape &sp) : s(sp.s) {}
  ~Shape() {}
  
  Shape& operator=(const Shape& new_n)
  {
    
    if(new_n.tag == TRIANGLE) {
      s.~Sphere();
      new (&t) Triangle(new_n.t);
    }
    else {
      t.~Triangle();
      new (&s) Sphere(new_n.s);
    }
    return *this;
  } 
  
  ShapeType tag;
  union {
    Sphere s;
    Triangle t;
  };
  
};

struct ObjEval{
  
  inline bool compare(Sphere a, Sphere b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Triangle a, Sphere b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Sphere a, Triangle b) { return a.getMorton() < b.getMorton(); }
  inline bool compare(Triangle a, Triangle b) { return a.getMorton() < b.getMorton(); }
  
    
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


#endif /* Shape_HH_INCLUDE */
