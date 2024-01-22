#pragma once
#include "data_structure.h"
#include <stdio.h>

class MarchingCube {

private :
	std::vector<Particle> particles;
	std::vector<Triangle> triangles;
	vec3 minVertex;
	vec3 maxVertex;

public:
	MarchingCube();
	~MarchingCube();

	bool get_vertices_by_txt(std::string filepath);
	bool make_polygon_with_particles();
	void make_polygon_with_particles(std::vector<vec3> vertices);
	bool make_grid();
	bool find_grid_minmax();
	void compute_vertex_density();
	void print_txt(std::string filepath);
};