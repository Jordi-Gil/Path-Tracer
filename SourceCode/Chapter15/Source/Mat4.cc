#include "Mat4.hh"


mat4::mat4(float val){
	for (int i = 0; i < 4; i++){
			m[i][0] = val;
			m[i][1] = val;
			m[i][2] = val;
			m[i][3] = val;
		}
	}
}

mat4(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4, float c1, float c2, float c3, float c4, float d1, float d2, float d3, float d4) {
	for(int i = 0; i < 4; i++) {
		if(i == 0) {
			m[i][0] = a1;
			m[i][1] = a2;
			m[i][2] = a3;
			m[i][3] = a4;
		}
		else if (i == 1) {
			m[i][0] = b1;
			m[i][1] = b2;
			m[i][2] = b3;
			m[i][3] = b4;			
		}
		else if (i == 2) {
			m[i][0] = c1;
			m[i][1] = c2;
			m[i][2] = c3;
			m[i][3] = c4;			
		}
		else {
			m[i][0] = d1;
			m[i][1] = d2;
			m[i][2] = d3;
			m[i][3] = d4;
		}
	}
}

mat4(float matrix[4][4]){
	for (int i = 0; i < 4; i++){
		m[i][0] = matrix[i][0];
		m[i][1] = matrix[i][1];
		m[i][2] = matrix[i][2];
		m[i][3] = matrix[i][3];
	} 
}
	
float operator[][](int i, int j) const{
	return m[i][j];
};

mat4 mat4::identity(){
	return mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
}
