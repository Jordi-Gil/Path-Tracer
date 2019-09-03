#ifndef BVH_node_HH_INCLUDE
#define BVH_node_HH_INCLUDE

#include "Hitable.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>

template<typename T>
struct lessX : public thrust::binary_function<T,T,bool>
{
	__device__ bool operator()(const T &lhs, const T &rhs) const {
		aabb box_left, box_right;
		
		Hitable *lh = *(Hitable**)lhs;
		Hitable *rh = *(Hitable**)rhs;
		
		if(!lh->bounding_box(0,0,box_left) || !rh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
		if(box_left.min().x() - box_right.min().x() < 0.0) return false;
		else return true;
	}
};

template<typename T>
struct lessY : public thrust::binary_function<T,T,bool>
{
	__device__ bool operator()(const T &lhs, const T &rhs) const {
		aabb box_left, box_right;
		
		Hitable *lh = *(Hitable**)lhs;
		Hitable *rh = *(Hitable**)rhs;
		
		if(!lh->bounding_box(0,0,box_left) || !rh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
		if(box_left.min().y() - box_right.min().y() < 0.0) return false;
		else return true;
	}
};

template<typename T>
struct lessZ : public thrust::binary_function<T,T,bool>
{
	__device__ bool operator()(const T &lhs, const T &rhs) const {
		aabb box_left, box_right;
		
		Hitable *lh = *(Hitable**)lhs;
		Hitable *rh = *(Hitable**)rhs;
		
		if(!lh->bounding_box(0,0,box_left) || !rh->bounding_box(0,0, box_right)) printf("No bouding box in BVH_node constructor\n");
		if(box_left.min().z() - box_right.min().z() < 0.0) return false;
		else return true;
	}
};

class BVH_node: public Hitable {
public:
  __device__ BVH_node() {}
  __device__ BVH_node(Hitable **list, int n, float time0, float time1, curandState *random, int depth, char *hint);
  
  __device__ virtual bool hit(const Ray& r, float tmin, float tmax, hit_record& rec) const;
  __device__ virtual bool bounding_box(float t0, float t1, aabb &box) const;
  __host__ __device__ virtual int length() const;
  
  Hitable *left;  // Left child
  Hitable *right; // Right child
  
  aabb box;
  
};


#endif /* BVH_node_HH_INCLUDE */
