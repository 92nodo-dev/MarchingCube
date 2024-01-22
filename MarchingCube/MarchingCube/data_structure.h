#pragma once
#include <vector>
#include <string>

struct vec3 {
	float x;
	float y;
	float z;

	vec3 operator+(vec3 v1) {
		return vec3{ x + v1.x, y + v1.y, z + v1.z };
	}
	vec3 operator-(vec3 v1) {
		return vec3{ x - v1.x, y - v1.y, z - v1.z };
	}
};

struct Particle {
	vec3 position;
	float density;
};

struct Cell {
	vec3 vertex[8];
};

struct Triangle {
	vec3 t1;
	vec3 t2;
	vec3 t3;
};

struct Grid {

};