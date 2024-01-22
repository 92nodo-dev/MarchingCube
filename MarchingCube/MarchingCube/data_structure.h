#pragma once
#include <vector>
#include <string>

struct vec3 {
	float x;
	float y;
	float z;
};

struct Particle {
	vec3 position;
	float density;
};

struct Triangle {
	vec3 t1;
	vec3 t2;
	vec3 t3;
};

struct Grid {

};