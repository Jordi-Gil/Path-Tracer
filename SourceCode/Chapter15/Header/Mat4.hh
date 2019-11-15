#ifndef MAT4_HH_INCLUDE
#define MAT4_HH_INCLUDE


class mat4 {

public:
	
	mat4() {}
	mat4(float val);
	mat4(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4, 
		float c1, float c2, float c3, float c4, float d1, float d2, float d3, float d4
	);
	mat4(float m[4][4]);
	
	float operator[][](int i, int j) const;
	
	static mat4 identity();
    
private:    
  
	float mat[4][4];
	
};
