#include "Node.hh"
#include <iostream>

Node::Node() {
  obj = NULL;
  left = NULL;
  right = NULL;
  parent = NULL;
}

bool Node::intersect(const Ray& r, float tmin, float tmax, hit_record& rec) {
  if(box.hit(r, tmin, tmax)) {
    hit_record left_rec, right_rec;

    bool hit_left;
    bool hit_right;

    if(left->obj) hit_left = left->obj->hit(r, tmin, tmax, left_rec);
    else hit_left = left->intersect(r, tmin, tmax, left_rec);
    
    if(right->obj) hit_right = right->obj->hit(r, tmin, tmax, right_rec);
    else hit_right = right->intersect(r, tmin, tmax, right_rec);
    
    if(hit_left && hit_right) {

      if(left_rec.t < right_rec.t) rec = left_rec;
      else rec = right_rec;

      return true;
  
    }
    else if (hit_left) {
        rec = left_rec;
        return true;
    }
    else if(hit_right) {
      rec = right_rec;
      return true;
    }
    else return false;
  }
  else {
		//std::cout << "False" << std::endl;
		return false;
	}
//   float t = tmax;
//   
//   Node *stack[STACK_SIZE];
//   int top = -1;
//   
//   stack[++top] = this;
//   
//   hit_record rec_left, rec_right;
//   bool hit_left = false, hit_right = false;
//   
//   do {
//   
//     hit_record rec_aux;
//     
//     Node *new_node = stack[top--];
//     
//     if(new_node->box.hit(r, tmin, tmax)) {
//       
//       if(new_node->isLeaf) {
//           bool hit = (new_node->obj)->hit(r, tmin, tmax, rec_aux);
//           if(hit && rec_aux.t < t){
//             
//             t = rec_aux.t;
//             
//             if(new_node->isRight){
//               rec_right = rec_aux;
//               hit_right = true;
//             }
//             else if(new_node->isLeft){
//               rec_left = rec_aux;
//               hit_left = true;
//             }
//             
//           }
//         }
//         
//       else {
//         
//         stack[++top] = new_node->right;
//         stack[++top] = new_node->left;
//         
//       }
//     }
//     
//   } while(top != -1);
//   
//   if(hit_left && hit_right) {
//     if(rec_left.t < rec_right.t) rec = rec_left;
//     else rec = rec_right;
//     return true;
//   }
//   else if(hit_left){ 
//     rec = rec_left; 
//     return true;
//   }
//   else if(hit_right) {
//     rec = rec_right;
//     return true;
//   }
//   else return false;
  
}

