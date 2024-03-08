#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "data_structure.h"

#include "MarchingCube.cuh"

namespace MarchingCube {

class MarchingCube {

private:
	//std::vector<Particle> particles;
	Particle* particles;

	DeviceData d_data;
	HostData h_data;
	/*
	Triangle* h_triangles;
	Triangle* d_triangles;
	Cell*** cells;
	Cell* d_cells;
	int* d_edgeTable;
	short int* d_triangleTable;
	*/

	int particleSize = 0;
	int triangleSize = 0;

	vec3 minVertex;
	vec3 maxVertex;
	float gridSize;
	int axisX;
	int axisY;
	int axisZ;

public:
	MarchingCube() {};
	~MarchingCube() {};

	bool get_vertices_by_txt(std::string filepath, std::string densityPath);
	bool get_vertices_by_vtk(std::string filepath);

	bool make_polygon_with_particles(float isoValue);
	bool make_polygon_with_particles(std::vector<vec3> vertices, float isoValue);
	bool generate_grid();
	bool find_grid_minmax();
	bool initialize_cell();
	bool put_density_into_cell();
	void make_triangle_arr();
	void flatten_cell_density();

	void allocate_particles(int size);
	Particle* get_particles() { return particles; }

	void compute_cell_bit(float isoValue);
	void alloc_device_memory();

	void make_triangle(float isoValue);

	void free_device_memory();

	void set_density();
	void set_vertices();

	//void compute_vertex_density();
	void print_txt(std::string filepath);
	void print_vtu(std::string filepath);
	void print_vtk(std::string filepath);

	void write_binary(std::string txt);
};

}