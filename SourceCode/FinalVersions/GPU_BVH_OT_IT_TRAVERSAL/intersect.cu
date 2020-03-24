__device__ bool intersect(Node *root, const Ray &cur_ray, float t_min, float t_max, hit_record &rec) {
  
  float t = FLT_MAX;
  
  Node *stack[STACK_SIZE];
  int top = -1;
  
  stack[++top] = root;
  bool intersected = false;
  
  
  hit_record rec_aux;
  
  do {
    
    Node *new_node = stack[top--];
    
    if(new_node->box.hit(cur_ray, t_min, t_max)) {
      
      if(new_node->isLeaf) {
        
        if( (new_node->obj)->hit(cur_ray, t_min, t_max, rec_aux) ) {
          
          if(rec_aux.t < t) {
            t = rec_aux.t;
            rec = rec_aux;
            intersected = true;
          }
          
        }
        
      }
      else {
        
        stack[++top] = new_node->right;
        stack[++top] = new_node->left;
        
      }
      
    }
    
  } while(top != -1);
  
  return intersected;
  
}

__device__ Vector3 color(const Ray& ray, Node *world, int depth, bool light, bool skybox, curandState *random, Skybox *sky, bool oneTex, unsigned char **d_textures){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3::One();
  for(int i = 0; i < depth; i++){ 
    hit_record rec;
    if( intersect(world, cur_ray, 0.00001, FLT_MAX, rec) ) {
//     if( world->checkCollision(cur_ray, 0.00001, FLT_MAX, rec) ) {
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted(rec.u, rec.v, oneTex, d_textures);
      
      if(rec.mat_ptr.scatter(cur_ray, rec, attenuation, scattered, random, oneTex, d_textures)){
        cur_attenuation *= attenuation;
        cur_attenuation += emitted;
        cur_ray = scattered;
      }
      else return cur_attenuation * emitted;
    }
    else {
      if(skybox && sky->hit(cur_ray, 0.00001, FLT_MAX, rec)){
        return cur_attenuation * rec.mat_ptr.emitted(rec.u, rec.v, oneTex, d_textures);
      }
      else {
        if(light) {
          Vector3 unit_direction = unit_vector(cur_ray.direction());
          float t = 0.5*(unit_direction.y() + 1.0);
          Vector3 c = (1.0 - t)*Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
          return cur_attenuation * c;
        }
        else return Vector3::Zero();
      }
    }
  }
  return Vector3::Zero();
}
