#include <iostream>
#include "Vector3.hh"

int main()
{

  int nx = 1920;
  int ny = 1080;

  std::cout << "P3\n" << nx << " " <<  ny << "\n255" << std::endl;
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
      
      Vector3 col(float(i) / float(nx), float(j) / float(ny), 0.2);
      
      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      std::cout << ir << " " << ig << " " << ib << std::endl;

    }
  }
}