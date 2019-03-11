#include <iostream>
#include <cfloat>
#include "Sphere.hh"
#include "HitableList.hh"
#include "Camera.hh"

Vector3 random_in_unit_sphere(){
    Vector3 p;
    do{
        p = 2.0*Vector3((rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0))) - Vector3::One();
    }while(p.squared_length() >= 1.0);
    return p;
}

Vector3 color(const Ray& ray, Hitable *world){
    hit_record rec;
    if(world->hit(ray, 0.001, MAXFLOAT, rec)){
        Vector3 target = rec.point + rec.normal + random_in_unit_sphere();
        return 0.5*color(Ray(rec.point, target-rec.point), world);
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
  int ns = 1000;

  std::cout << "P3\n" << nx << " " <<  ny << "\n255" << std::endl;
  
  Hitable *list[2];
  list[0] = new Sphere(Vector3(0, 0, -1), 0.5);
  list[1] = new Sphere(Vector3(0, -100.5, -1), 100);
  
  Hitable *world = new HitableList(list, 2);
  
  Camera cam;
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
        
      Vector3 col = Vector3::Zero();
      
      for(int s = 0; s < ns; s++){
        float u = float(i + (rand()/(RAND_MAX + 1.0))) / float(nx);
        float v = float(j + (rand()/(RAND_MAX + 1.0))) / float(ny);
        
        Ray r = cam.get_ray(u, v);
        //Vector3 p = r.point_at_parameter(2.0);
        
        col += color(r, world);
      }
      
      
      col /= float(ns);
      col = Vector3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << std::endl;

    }
  }
}
