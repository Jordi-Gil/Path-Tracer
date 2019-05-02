#include <iostream>
#include <cfloat>
#include "Sphere.hh"
#include "HitableList.hh"
#include "Camera.hh"
#include "Material.hh"

Vector3 color(const Ray& ray, Hitable *world, int depth){
    hit_record rec;
    if(world->hit(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        if(depth < 500 && rec.mat_ptr->scatter(ray, rec, attenuation, scattered)){
            return attenuation*color(scattered, world, depth+1);
        }
        else return Vector3::Zero();
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
  
  Hitable *list[5];
  list[0] = new Sphere(Vector3(0, 0, -1), 0.5, new Lambertian(Vector3(0.1, 0.2, 0.5)));
  list[1] = new Sphere(Vector3(0, -100.5, -1), 100, new Lambertian(Vector3(0.8, 0.8, 0.0)));
  list[2] = new Sphere(Vector3(1, 0, -1), 0.5, new Metal(Vector3(0.8,0.6,0.2),0.3));
  list[3] = new Sphere(Vector3(-1, 0, -1), 0.5, new Dielectric(1.5));
  list[4] = new Sphere(Vector3(-1, 0, -1), -0.45, new Dielectric(1.5));
  
  Hitable *world = new HitableList(list, 5);
  
  Vector3 lookfrom(3,3,2);
  Vector3 lookat(0,0,-1);
  Vector3 up(0,1,0);
  float dist_to_focus = (lookfrom-lookat).length();
  float aperture = 2.0;
  
  Camera cam(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus);
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
        
      Vector3 col = Vector3::Zero();
      
      for(int s = 0; s < ns; s++){
        float u = float(i + (rand()/(RAND_MAX + 1.0))) / float(nx);
        float v = float(j + (rand()/(RAND_MAX + 1.0))) / float(ny);
        
        Ray r = cam.get_ray(u, v);
        //Vector3 p = r.point_at_parameter(2.0);
        
        col += color(r, world, 0);
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
