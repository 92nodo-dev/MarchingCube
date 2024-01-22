#pragma once
#include "data_structure.h"
#include <stdio.h>

class MarchingCube {
private :
	std::vector<vec3> vertices;
	std::vector<Triangle> triangles;
public:
	MarchingCube();
	~MarchingCube();

	bool get_vertices_by_txt(std::string filepath);
	void make_marchingCube_with_vertices(std::vector<vec3> vertices);
	void print_txt(std::string filepath);
};