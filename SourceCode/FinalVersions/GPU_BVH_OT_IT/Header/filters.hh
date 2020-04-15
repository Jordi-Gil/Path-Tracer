#include "Vector3.cuh"

struct Vector3eval {
    __host__ __device__ inline bool operator()(const Vector3 &a, const Vector3 &b){
      return (a[0] < b[0] && a[1] < b[1] && a[2] < b[2]);
    }
    
};

int getColor255(unsigned char *image, int nx, int i, int j, int offset){
  return image[i*nx*3 + j*3 + offset];
}

Vector3 *getWindow(unsigned char *image, int i, int j, int nx) {
  Vector3 *window;
  
  window = new Vector3[8];
  
  window[0] = Vector3(getColor255(image, nx, i-1, j-1, 0),getColor255(image, nx, i-1, j-1, 1),getColor255(image, nx, i-1, j-1, 2));
  window[1] = Vector3(getColor255(image, nx, i-1, j  , 0),getColor255(image, nx, i-1, j  , 1),getColor255(image, nx, i-1, j  , 2));
  window[2] = Vector3(getColor255(image, nx, i-1, j+1, 0),getColor255(image, nx, i-1, j+1, 1),getColor255(image, nx, i-1, j+1, 2));
  window[3] = Vector3(getColor255(image, nx, i  , j-1, 0),getColor255(image, nx, i  , j-1, 1),getColor255(image, nx, i  , j-1, 2));
  window[4] = Vector3(getColor255(image, nx, i  , j  , 0),getColor255(image, nx, i  , j  , 1),getColor255(image, nx, i  , j  , 2));
  window[5] = Vector3(getColor255(image, nx, i  , j+1, 0),getColor255(image, nx, i  , j+1, 1),getColor255(image, nx, i  , j+1, 2));
  window[6] = Vector3(getColor255(image, nx, i+1, j-1, 0),getColor255(image, nx, i+1, j-1, 1),getColor255(image, nx, i+1, j-1, 2));
  window[7] = Vector3(getColor255(image, nx, i+1, j  , 0),getColor255(image, nx, i+1, j  , 1),getColor255(image, nx, i+1, j  , 2));
  window[8] = Vector3(getColor255(image, nx, i+1, j+1, 0),getColor255(image, nx, i+1, j+1, 1),getColor255(image, nx, i+1, j+1, 2));
  
  std::sort(window, window+8, Vector3eval());
  
  return window;
}

Vector3 lab2xyz(Vector3 color){
  
  float refX = 95.047f;
  float refY = 100.000f;
  float refZ = 108.883f;
  
  float fy = (color[0]+16)/116;
  float fx = (color[1]/500)+fy;
  float fz = fy - (color[2]/200);
  
  float x,y,z;
  
  if(pow(fx,3) > (216/24389)) x = pow(fx,3);
  else x = ((116*fx)-16)/(24389/27);
  
  if(color[0] > (216/27)) y = pow(fy,3);
  else y = color[0]/(24389/27);
  
  if(pow(fz,3) > (216/24389)) z = pow(fz,3);
  else z = ((116*fz)-16)/(24389/27);
  
  x *= refX;
  y *= refY;
  z *= refZ;
  
  return Vector3(x,y,z);
  
}

Vector3 lab2rgb(Vector3 color){
  
  Vector3 xyzColor = lab2xyz(color);
  
  float x = xyzColor[0]/100.0f;
  float y = xyzColor[1]/100.0f;
  float z = xyzColor[2]/100.0f;
  
  float r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314;
  float g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560;
  float b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252;
  
  if(r > 0.0031308) r = (1.055 * pow(r,1.0f/2.4f)) - 0.055;
  else r = r*12.92;
  
  if(g > 0.0031308) g = (1.055 * pow(g,1.0f/2.4f)) - 0.055;
  else g = g*12.92;
  
  if(b > 0.0031308) b = (1.055 * pow(b,1.0f/2.4f)) - 0.055;
  else b = b*12.92;
  
  int rN = int(255.99f*r);
  int gN = int(255.99f*g);
  int bN = int(255.99f*b);
  
  return Vector3(rN,gN,bN);
}

Vector3 rgb2xyz(Vector3 color){
  
  float r = color[0]/255.0f;
  float g = color[1]/255.0f;
  float b = color[2]/255.0f;
  
  if(r > 0.04045) r = pow( ( (r + 0.055) / 1.055 ), 2.4);
  else r = r/12.92;
  
  if(g > 0.04045) g = pow( ( (g + 0.055) / 1.055 ), 2.4);
  else g = g/12.92;
  
  if(b > 0.04045) b = pow( ( (b + 0.055) / 1.055 ), 2.4);
  else b = b/12.92;
  
  r *= 100.0f;
  g *= 100.0f;
  b *= 100.0f;
  
  float x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  float y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
  float z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
  
  return Vector3(x,y,z);
  
}

Vector3 rgb2lab(Vector3 color){
  
  Vector3 xyzColor = rgb2xyz(color);
  //std::cout << "XYZ: " << xyzColor << std::endl;
  
  float refX = 95.047f;
  float refY = 100.000f;
  float refZ = 108.883f;
  
  float x = xyzColor[0] / refX;
  float y = xyzColor[1] / refY;
  float z = xyzColor[2] / refZ;
  
  if(x > (216/24389)) 
    x = std::cbrt(x);
  else
    x = (((24389/27)*x) + 16)/116; //(7.787 * x) + (16/116);
  
  if(y > (216/24389))
    y = std::cbrt(y);
  else y = (((24389/27)*y) + 16)/116; //(7.787 * y) + (16/116);
  
  if(z > (216/24389))
    z = std::cbrt(z);
  else z = (((24389/27)*z) + 16)/116; //(7.787 * z) + (16/116);
  
  float a = 500 * (x-y);
  float l = (116 * y) - 16;
  float b = 200 * (y-z);
  
  return Vector3(l,a,b);
  
}

float distance(Vector3 a, Vector3 b){
  
  return float( (a[0] - b[0])*(a[0] - b[0]) +
                (a[1] - b[1])*(a[1] - b[1]) +
                (a[2] - b[2])*(a[2] - b[2])
  );
  
}

float gaussian(int i, int j, int k, int l, float dist, float gd, float gr){
  float a = (i-k)*(i-k);
  float b = (j-l)*(j-l);
  float c = a + b;
  return std::exp(-(c/gd) - (dist/gr));
}

Vector3 applyFilter(int i, int j, Vector3 color, int diameter, float gd, float gr, int nx, int ny, unsigned char *image){
  
  int half = diameter/2;
  Vector3 clabColor = rgb2lab(color);
  
  Vector3 wp = Vector3::Zero();
  float norm = 0;
  
  for(int l = 0; l < diameter; l++){
    for(int k = 0; k < diameter; k++){
      int neighbour_x = i - (half - l);
      int neighbour_y = j - (half - k);
      if(neighbour_x >= 0 && neighbour_y >= 0 && neighbour_x < ny && neighbour_y < nx) {
        Vector3 colorN = rgb2lab(Vector3(getColor255(image, nx, neighbour_x, neighbour_y, 0),getColor255(image, nx, neighbour_x, neighbour_y, 1),getColor255(image, nx, neighbour_x, neighbour_y, 2)));
        float g = gaussian(i,j,neighbour_x,neighbour_y,distance(clabColor, colorN),2*(gd*gd),2*(gr*gr));
        norm += g;
        wp += (colorN*g);
      }
    }
  }
  
  Vector3 colorFiltered = lab2rgb(wp/norm) ;
  
  return colorFiltered;
  
}

void bilateralFilter(int diameter, int nx, int ny, unsigned char *image, unsigned char *imageFiltered, float sd, float sr){
  
  for(int i = 0; i < ny; i++){
    for(int j = 0; j < nx; j++){

      
      Vector3 color = applyFilter(i, j, Vector3(getColor255(image, nx, i, j, 0),getColor255(image, nx, i, j, 1),getColor255(image, nx, i, j, 2)), diameter, sd, sr, nx, ny, image);
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0]; 
      imageFiltered[i*nx*3 + j*3 + 1] = color[1];
      imageFiltered[i*nx*3 + j*3 + 2] = color[2];
      
    }
  }
}

void medianFilter(int nx, int ny, unsigned char *image, unsigned char *imageFiltered){
  
  for(int i = 1; i < ny-1; ++i){
    for(int j = 1; j < nx-1; ++j){
      Vector3 *window = getWindow(image, i, j, nx);
      
      Vector3 color = window[4];
      delete window;
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0]; 
      imageFiltered[i*nx*3 + j*3 + 1] = color[1];
      imageFiltered[i*nx*3 + j*3 + 2] = color[2];
      
    }
  }
}

void meanFilter(int nx, int ny, unsigned char *image, unsigned char *imageFiltered){
  
  for(int i = 1; i < ny-1; ++i){
    for(int j = 1; j < nx-1; ++j){
      Vector3 *window = getWindow(image, i, j, nx);
      
      Vector3 color = window[0];
      for(int k = 1; k < 8; k++ ) color += window[k];
      delete window;
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0]/9.0; 
      imageFiltered[i*nx*3 + j*3 + 1] = color[1]/9.0;
      imageFiltered[i*nx*3 + j*3 + 2] = color[2]/9.0;
      
    }
  }
}
