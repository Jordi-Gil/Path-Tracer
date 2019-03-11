#include <iostream>
#include "Ray.hh"

bool hit_sphere(const Vector3& center, float radius, const Ray& ray){
    Vector3 oc = ray.origin() - center;
    float a = dot(ray.direction(), ray.direction());
    float b = 2.0 * dot(oc, ray.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

Vector3 color(const Ray& ray){
    
    if(hit_sphere(Vector3(0, 0, -1), 0.5, ray)){
        return Vector3(1, 0, 0);
    }
    Vector3 unit_direction = unit_vector(ray.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t) * Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
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
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
      
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);
      
      Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
      Vector3 col = color(r);
      
      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << std::endl;

    }
  }
}
