#include "BVH_node.cuh"

/*
__device__ int box_x_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
  
  if(box_left.min().x() - box_right.min().x() < 0.0) return -1;
  else return 1;
  
}

__device__ int box_y_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
  if(box_left.min().y() - box_right.min().y() < 0.0) return -1;
  else return 1;
  
}

__device__ int box_z_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
  if(box_left.min().z() - box_right.min().z() < 0.0) return -1;
  else return 1;
  
}
*/

__device__ BVH_node::BVH_node(Hitable **list, int n, float time0, float time1, curandState *random, int depth, char *hint) {

	printf("Side: %s, Prof: %d, address: %p \n", hint, depth, &list);
	int axis = int(3*curand_uniform(random));
/*
	if (axis == 0) thrust::sort(list, list + n*sizeof(Hitable *), lessX<Hitable*>());
	else if (axis == 1) thrust::sort(list, list + n*sizeof(Hitable *), lessY<Hitable*>());
	else thrust::sort(list, list + n*sizeof(Hitable *), lessZ<Hitable*>());
*/
  
	if(n == 1) {
		left = right = list[0];
	}
	else if (n == 2) {
		left = list[0];
		right = list[1];
	}
	else {
		left = new BVH_node(list, n/2, time0, time1, random, depth+1,"Left");
		right = new BVH_node(list + n/2, n - n/2, time0, time1, random, depth+1,"Right"); 
	}
  
	aabb box_left, box_right;
	if(!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right)) 
		printf("No bouding box in BVH_node constructor\n");
  
	box = surrounding_box(box_left, box_right);
  
}

__host__ __device__ int BVH_node::length() const {
    if(!left && !right) return 1;
    else {
      int left_count = 0;
      int right_count = 0;
      
      left_count += left->length();
      right_count += right->length();
      return max(left_count,right_count);
    }
}

__device__ bool BVH_node::hit(const Ray& r, float tmin, float tmax, hit_record& rec) const {

	if(box.hit(r, tmin, tmax)) {
		hit_record left_rec, right_rec;
		bool hit_left = left->hit(r, tmin, tmax, left_rec);
		bool hit_right = right->hit(r, tmin, tmax, right_rec);
    
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
	else return false;
}

__device__ bool BVH_node::bounding_box(float t0, float t1, aabb& b) const {
	
	t0 = t0;
	t1 = t1;
	b = box;
	return true;
  
}
