#include "Vector3.hh"

struct Vector3eval {
  inline bool operator()(const Vector3 &a, const Vector3 &b){
    return (a[0] < b[0] && a[1] < b[1] && a[2] < b[2]);
  }
};

int getColor255(unsigned char *image, int nx, int i, int j, int offset){
  return image[i*nx*3 + j*3 + offset];
}

Vector3 lab2xyz(Vector3 color){
  
  float refX = 95.047f;
  float refY = 100.000f;
  float refZ = 108.883f;
  
  float fy = (color[0]+16)/116;
  float fx = (color[1]/500)+fy;
  float fz = fy - (color[2]/200);
  
  float delta = (6.f/29.f)*(6.f/29.f);
  
  float x,y,z;
  
  if(fx > (6.f/29.f)) x = pow(fx,3);
  else x = 3*delta*(fx - (4.f/29.f));
  
  if(fy > (6.f/29.f)) y = pow(fy,3);
  else y = 3*delta*(fy - (4.f/29.f));
  
  if(fz > (6.f/29.f)) z = pow(fz,3);
  else z = 3*delta*(fz - (4.f/29.f));
  
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
  
  float refX = 95.047f;
  float refY = 100.000f;
  float refZ = 108.883f;
  
  float x = xyzColor[0] / refX;
  float y = xyzColor[1] / refY;
  float z = xyzColor[2] / refZ;
  
  float delta = (6.f/29.f)*(6.f/29.f);
  
  if(x > (delta * (6.f/29.f))) 
    x = std::cbrt(x);
  else
    x = (x / (3*delta)) + (4.f/29.f);
  
  if(y > (delta * (6.f/29.f))) 
    y = std::cbrt(y);
  else 
    y = (y / (3*delta)) + (4.f/29.f);
  
  if(z > (delta * (6.f/29.f))) 
    z = std::cbrt(z);
  else 
    z = (z / (3*delta)) + (4.f/29.f);
  
  float a = 500 * (x-y);
  float l = (116 * y) - 16;
  float b = 200 * (y-z);
  
  return Vector3(l,a,b);
  
}

float distance(int i, int j, int k, int l){
  
  int x2 = (k-i)*(k-i);
  int y2 = (l-j)*(l-j);
  
  return sqrt( x2 + y2 );
}

float gaussian(float dist, float sigma){
  float a = dist*dist;
  float b = sigma*sigma;
  return std::exp(-a/(2*b));
}

Vector3 bilateralConvolution(int i, int j, Vector3 color, int diameter, float sigmaD, float sigmaR, int nx, int ny, unsigned char *image){
  
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
        
        float gd = gaussian(distance(i,j, neighbour_x, neighbour_y), sigmaD);
        float gr = gaussian((colorN - clabColor).length(), sigmaR);
        
        norm += (gd*gr);
        wp += (gd*gr)*colorN;
      }
    }
  }
  
  Vector3 colorFiltered = lab2rgb(wp/norm) ;
  
  return colorFiltered;
  
}

void bilateralFilter(int diameter, int nx, int ny, unsigned char *image, unsigned char *imageFiltered, float sd, float sr){
  
  for(int i = 0; i < ny; i++){
    for(int j = 0; j < nx; j++){

      
      Vector3 color = bilateralConvolution(i, j, Vector3(getColor255(image, nx, i, j, 0),getColor255(image, nx, i, j, 1),getColor255(image, nx, i, j, 2)), diameter, sd, sr, nx, ny, image);
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0]; 
      imageFiltered[i*nx*3 + j*3 + 1] = color[1];
      imageFiltered[i*nx*3 + j*3 + 2] = color[2];
      
    }
  }
}

Vector3 *medianConvolution(int diameter, int i, int j, int nx, int ny, unsigned char *image){
  
  int half = diameter/2;
  Vector3 *color = new Vector3[diameter*diameter];
  
  int count = 0;
  
  for(int k = 0; k < diameter; k++) {
    for(int l = 0; l < diameter; l++) {
      
      int neighbour_x = i - (half - l);
      int neighbour_y = j - (half - k);

      color[count++] = Vector3(getColor255(image, nx, neighbour_x, neighbour_y, 0),getColor255(image, nx, neighbour_x, neighbour_y, 1),getColor255(image, nx, neighbour_x, neighbour_y, 2));
      
    }
  }
  
  std::sort(color, color+(diameter*diameter), Vector3eval());
  
  return color;
}

void medianFilter(int diameter, int nx, int ny, unsigned char *image, unsigned char *imageFiltered){
  
  int half = diameter/2;
  
  for(int i = half; i < ny-half; ++i){
    for(int j = half; j < nx-half; ++j){
      
      Vector3 *window = medianConvolution(diameter, i, j, nx, ny, image);
      
      Vector3 color = window[(diameter*diameter)/2];
      delete window;
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0]; 
      imageFiltered[i*nx*3 + j*3 + 1] = color[1];
      imageFiltered[i*nx*3 + j*3 + 2] = color[2];
      
    }
  }
}

Vector3 meanConvolution(int diameter, int i, int j, int nx, int ny, unsigned char *image){
  
  int half = diameter/2;
  Vector3 color = Vector3::Zero();
  
  for(int k = 0; k < diameter; k++) {
    for(int l = 0; l < diameter; l++) {
      
      int neighbour_x = i - (half - l);
      int neighbour_y = j - (half - k);
      
      if(neighbour_x >= 0 && neighbour_y >= 0 && neighbour_x < ny && neighbour_y < nx) 
        color += Vector3(getColor255(image, nx, i, j, 0),getColor255(image, nx, i, j, 1),getColor255(image, nx, i, j, 2));
      
    }
  }
  return color/(diameter*diameter);
}

void meanFilter(int diameter, int nx, int ny, unsigned char *image, unsigned char *imageFiltered){
  
  int half = diameter/2;
  
  for(int i = half; i < ny-half; ++i){
    for(int j = half; j < nx-half; ++j){
      
      Vector3 color = meanConvolution(diameter, i, j, nx, ny, image);
      
      imageFiltered[i*nx*3 + j*3 + 0] = color[0];
      imageFiltered[i*nx*3 + j*3 + 1] = color[1];
      imageFiltered[i*nx*3 + j*3 + 2] = color[2];
      
    }
  }
}
