#pragma once
#include <stdio.h>
#include "data_structure.h"

#include "MarchingCube.cuh"

class MarchingCube {

private :
	std::vector<Particle> particles;
	std::vector<Triangle> triangles;
	Cell*** cells;

	vec3 minVertex;
	vec3 maxVertex;

	float gridSize;
	int axisX;
	int axisY;
	int axisZ;

public:
	MarchingCube() {};
	~MarchingCube() {};

	bool get_vertices_by_txt(std::string filepath);
	bool make_polygon_with_particles();
	bool make_polygon_with_particles(std::vector<vec3> vertices);
	bool generate_grid();
	bool find_grid_minmax();
	bool initialize_cell();
	bool put_density_into_cell();

	//void compute_vertex_density();
	void print_txt(std::string filepath);
};