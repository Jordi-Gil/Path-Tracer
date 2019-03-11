#include <iostream>
#include <cfloat>
#include "Sphere.hh"
#include "HitableList.hh"


Vector3 color(const Ray& ray, Hitable *world){
    hit_record rec;
    if(world->hit(ray, 0.001, MAXFLOAT, rec)){
        return 0.5*Vector3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else{
        Vector3 unit_direction = unit_vector(ray.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0-t) * Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
    }
}

int main()
{

  int nx = 2000;
  int ny = 1000;

  std::cout << "P3\n" << nx << " " <<  ny << "\n255" << std::endl;
  
  Vector3 lower_left_corner(-2.0, -1.0, -1.0);
  Vector3 horizontal(4.0, 0.0, 0.0);
  Vector3 vertical(0.0, 2.0, 0.0);
  Vector3 origin(0.0, 0.0, 0.0);
  
  Hitable *list[2];
  list[0] = new Sphere(Vector3(0, 0, -1), 0.5);
  list[1] = new Sphere(Vector3(0, -100.5, -1), 100);
  
  Hitable *world = new HitableList(list, 2);
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
      
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);
      
      Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
      
      //Vector3 p = r.point_at_parameter(2.0);
      Vector3 col = color(r, world);
      
      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << std::endl;

    }
  }
}
