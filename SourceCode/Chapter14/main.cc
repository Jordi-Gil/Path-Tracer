#include <iostream>
#include <cfloat>
#include <fstream>
#include <string>
#include <sys/time.h>

#include "Sphere.hh"
#include "MovingSphere.hh"
#include "HitableList.hh"
#include "Camera.hh"
#include "Material.hh"
#include "BVH_node.hh"

double getusec_() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}

#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME stamp = getusec_() - stamp;\
                        stamp = stamp/1e6;

void error(const char *message){
  std::cout << message << std::endl;
  exit(0);
}

void help(){
  std::cout << "\n"  << std::endl;
  std::cout << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
  std::cout << "\t                size: (1200x600) | AAit: 10 | depth: 50 | spheres: 11 | nthreads: 8"  << std::endl;
  std::cout << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cout << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cout << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cout << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cout << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cout << "\t[-f][--file]    File name of pic generated." << std::endl;
  std::cout << "\t[-h][--help]    Show help." << std::endl;
  std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Examples of usage:" << std::endl;
  std::cout << "./path_tracing_sec -d"  << std::endl;
  std::cout << "./path_tracing_sec -sizeX 2000"<< std::endl;
  exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &filename){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 512; ny = 512; ns = 50; depth = 50; dist = 11; filename = "spheres.ppm";
  
  bool v_default = false;
  
  for(int i = 1; i < argc; i += 2){
    
    if(v_default) error("Error usage. Use [-h] [--help] to see the usage.");
    
  	if (std::string(argv[i]) == "-d" || std::string(argv[i]) == "--default") {
  		if ((i+1) < argc) error("The default parameter cannot have more arguments.");
  		std::cerr << "Default\n";
  		v_default = true;
  	} else if (std::string(argv[i]) == "-sizeX"){
  		if ((i+1) >= argc) error("-sizeX value expected");
  		nx = atoi(argv[i+1]);
  		if(nx == 0) error("-sizeX value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-sizeY"){
  		if ((i+1) >= argc) error("-sizeY value expected");
  		ny = atoi(argv[i+1]);
  		if(ny == 0) error("-sizeY value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-AAit"){
  		if ((i+1) >= argc) error("-AAit value expected");
  		ns = atoi(argv[i+1]);
  		if(ns == 0) error("-AAit value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-depth"){
  		if ((i+1) >= argc) error("-depth value expected");
  		depth = atoi(argv[i+1]);
  		if(depth == 0) error("-depth value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-spheres"){
  		if ((i+1) >= argc) error("-spheres value expected");
  		dist = atoi(argv[i+1]);
  		if(depth == 0) error("-spheres value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
  		if ((i+1) >= argc) error("--file / -f value expected");
  		filename = std::string(argv[i+1]);
  		filename = filename+".ppm";
  	} else if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
  		help();
  	} else {
  		error("Error usage. Use [-h] [--help] to see the usage.");
  	}
  }
}

Hitable *random_scene(int dist){
  int n = (2*22)*(2*22)+4;
  Hitable **list = new Hitable*[n+1];
  list[0] = new Sphere(Vector3(0,-1000,0),1000, new Lambertian(Vector3(0.5,0.5,0.5)));
  int i = 1;
  for(int a = -dist; a < dist; a++){
    for(int b = -dist; b < dist; b++){
      float choose_mat = (rand()/(RAND_MAX + 1.0));
      Vector3 center(a+0.9*(rand()/(RAND_MAX + 1.0)), 0.2, b+0.9*(rand()/(RAND_MAX + 1.0)));
      
      if((center-Vector3(0,0,0)).length() > 0.995){
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
          list[i++] = new Sphere(center, 0.2, new Dielectric(Vector3::One(),1.5));
        }
      }
    }
  }
  
  list[i++] = new Sphere(Vector3(0,1,0), 1.0, new Dielectric(Vector3::One(),1.5));
  list[i++] = new Sphere(Vector3(-4,1,0),1.0, new Lambertian(Vector3(0.4,0.2,0.1)));
  list[i++] = new Sphere(Vector3(4,1,0),1.0, new Metal(Vector3(0.7,0.6,0.5),0.0));
  
  list[i++] = new Sphere(Vector3( 4, 1, 5), 1.0, new Metal(Vector3(0.9, 0.2, 0.2),0.0));
  
  return new BVH_node(list,i,0,1);
}

Vector3 color(const Ray& ray, Hitable *world, int depth, int const _depth){
    hit_record rec;
    if(world->hit(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        if(depth < _depth && rec.mat_ptr->scatter(ray, rec, attenuation, scattered)){
            return attenuation*color(scattered, world, depth+1, _depth);
        }
        else return Vector3::Zero();
    }
    else{
        Vector3 unit_direction = unit_vector(ray.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0-t) * Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
    }
}

int main(int argc, char **argv)
{
  
  int nx, ny, ns, depth, dist;
  std::string filename;
  
  parse_argv(argc, argv, nx, ny, ns, depth, dist, filename);
  
  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels" << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << n << " spheres." << std::endl;
  
  std::ofstream pic;
  pic.open(filename.c_str());
  
  pic << "P3\n" << nx << " " <<  ny << "\n255" << "\n";
  
  
  Hitable *world = random_scene(dist);
  
  Vector3 lookfrom(13,2,3);
  Vector3 lookat(0,0,0);
  float dist_to_focus = 10.0;
  float aperture = 0.1;

  Camera cam(lookfrom, lookat, Vector3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
  
  std::cout << "Creating image..." << std::endl;
  double stamp;
  START_COUNT_TIME;
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
        
      Vector3 col = Vector3::Zero();
      
      for(int s = 0; s < ns; s++){
        float u = float(i + (rand()/(RAND_MAX + 1.0))) / float(nx);
        float v = float(j + (rand()/(RAND_MAX + 1.0))) / float(ny);
        
        Ray r = cam.get_ray(u, v);
        
        col += color(r, world, 0, depth);
      }
      
      col /= float(ns);
      col = Vector3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      pic << ir << " " << ig << " " << ib << "\n";
    }
  }
  pic.close();
  STOP_COUNT_TIME;
  std::cout << "Image created in " << stamp << " seconds" << std::endl;
}


/*
 * int seconds, hours, minutes;
cin >> seconds;
minutes = seconds / 60;
hours = minutes / 60;
cout << seconds << " seconds is equivalent to " << int(hours) << " hours " << int(minutes%60) 
     << " minutes " << int(seconds%60) << " seconds.";
*/
