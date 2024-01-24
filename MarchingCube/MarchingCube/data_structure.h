#pragma once
#include <vector>
#include <string>

struct vec3 {
	float x;
	float y;
	float z;

	vec3 operator+(vec3 v1) { return vec3{ x + v1.x, y + v1.y, z + v1.z };}
	vec3 operator-(vec3 v1) { return vec3{ x - v1.x, y - v1.y, z - v1.z };}
};

struct Particle {
	vec3 position;
	float density;
};

struct Cell {
	float density;
	int particleCnt;
	int vertexCase;

	vec3 coordinate;
	vec3 vertex[8];
	Cell() 
	{ 
		particleCnt = 0; 
		density = 0;
	}
	void set_vertex_with_coordinate(float gridSize) 
	{
		vertex[0] = vec3{ coordinate.x - (gridSize * 0.5f), coordinate.y - (gridSize * 0.5f) , coordinate.z - (gridSize * 0.5f) };
		vertex[1] = vec3{ coordinate.x + (gridSize * 0.5f), coordinate.y - (gridSize * 0.5f) , coordinate.z - (gridSize * 0.5f) };
		vertex[2] = vec3{ coordinate.x + (gridSize * 0.5f), coordinate.y + (gridSize * 0.5f) , coordinate.z - (gridSize * 0.5f) };
		vertex[3] = vec3{ coordinate.x - (gridSize * 0.5f), coordinate.y + (gridSize * 0.5f) , coordinate.z - (gridSize * 0.5f) };
		vertex[4] = vec3{ coordinate.x - (gridSize * 0.5f), coordinate.y - (gridSize * 0.5f) , coordinate.z + (gridSize * 0.5f) };
		vertex[5] = vec3{ coordinate.x + (gridSize * 0.5f), coordinate.y - (gridSize * 0.5f) , coordinate.z + (gridSize * 0.5f) };
		vertex[6] = vec3{ coordinate.x + (gridSize * 0.5f), coordinate.y + (gridSize * 0.5f) , coordinate.z + (gridSize * 0.5f) };
		vertex[7] = vec3{ coordinate.x - (gridSize * 0.5f), coordinate.y + (gridSize * 0.5f) , coordinate.z + (gridSize * 0.5f) };
	}
};

struct Triangle {
	vec3 t1;
	vec3 t2;
	vec3 t3;
};

struct Grid {

};