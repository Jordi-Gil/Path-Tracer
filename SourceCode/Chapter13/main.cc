#include <iostream>
#include <cfloat>
#include <sys/time.h>

#include "Sphere.hh"
#include "MovingSphere.hh"
#include "HitableList.hh"
#include "Camera.hh"
#include "Material.hh"

double getusec_() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}

#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME stamp = getusec_() - stamp;\
                        stamp = stamp/1e6;

Hitable *random_scene(){
  
  
  int dist = 4;
  int n = (2*dist)*(2*dist) - 2 + 4 - 1;
  
  Hitable **list = new Hitable*[n+1];
  list[0] = new Sphere(Vector3(0,-1000,0),1000, new Lambertian(Vector3(0.5,0.5,0.5)));
  int i = 1;
  for(int a = -dist; a < dist; a++){
    for(int b = -dist; b < dist; b++){
      float choose_mat = (rand()/(RAND_MAX + 1.0));
      Vector3 center(a+0.9*(rand()/(RAND_MAX + 1.0)), 0.2, b+0.9*(rand()/(RAND_MAX + 1.0)));
      
      if((center-Vector3(4,0.2,0)).length() > 0.9){
        if(choose_mat < 0.8){ //diffuse
          list[i++] = new MovingSphere(center, center+Vector3(0,0.5*(rand()/(RAND_MAX + 1.0)),0), 0.0, 1.0, 0.2, 
            new Lambertian(Vector3(
                (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                (rand()/(RAND_MAX + 1.0)))));
        }
        else if(choose_mat < 0.95){ //metal
          list[i++] = new Sphere(center, 0.2, new Metal(Vector3(
            0.5*(1+(rand()/(RAND_MAX + 1.0))),
            0.5*(1+(rand()/(RAND_MAX + 1.0))),
            0.5*(1+(rand()/(RAND_MAX + 1.0)))
          ), 0.5*(rand()/(RAND_MAX + 1.0))));
        }
        else{
          list[i++] = new Sphere(center, (rand()/(RAND_MAX + 4.0)), new Dielectric(1.5));
        }
      }
    }
  }
  list[i++] = new Sphere(Vector3(0,1,0), 1.0, new Dielectric(1.5));
  list[i++] = new Sphere(Vector3(-4,1,0),1.0, new Lambertian(Vector3(0.4,0.2,0.1)));
  list[i++] = new Sphere(Vector3(4,1,0),1.0, new Metal(Vector3(0.7,0.6,0.5),0.0));
  
  return new HitableList(list, i);
}

Vector3 color(const Ray& ray, Hitable *world, int depth){
    
    hit_record rec;
    if(world->hit(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        if(depth < 50 && rec.mat_ptr->scatter(ray, rec, attenuation, scattered)){
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
  int ns = 50;

  std::cout << "P3\n" << nx << " " <<  ny << "\n255" << std::endl;
  
  
  Hitable *world = random_scene();
  
  Vector3 lookfrom(13,2,3);
  Vector3 lookat(0,0,0);
  float dist_to_focus = 10.0;
  float aperture = 0.1;

  Camera cam(lookfrom, lookat, Vector3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
  
  double stamp;
  START_COUNT_TIME;
  
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
  
  /* End timing  */
  STOP_COUNT_TIME;
  std::cerr << "Total execution time " << stamp << "seconds " << std::endl;
  
}
