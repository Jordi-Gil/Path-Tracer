#include "Camera.hh"

Camera::Camera(){
    lower_left_corner = Vector3(-2.0, -1.0, -1.0);
    horizontal = Vector3(4.0, 0.0, 0.0);
    vertical = Vector3(0.0, 2.0, 0.0);
    origin = Vector3::Zero();
}

Ray Camera::get_ray(float u, float v){
    return Ray(origin, 
               lower_left_corner + u*horizontal + v*vertical - origin);
}
