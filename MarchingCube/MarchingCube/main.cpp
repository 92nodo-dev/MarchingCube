#include "testor.h"
#include "MarchingCube.cuh"

int main()
{
	Testor myTestor;
	vec3 a = { 1.0,-5.0,3.0 };
	vec3 b = { -5.0,3.0,9.0 };
	vec3 c = a - b;

	std::vector<vec3> testVertex;
	testVertex.push_back(a);
	testVertex.push_back(b);
	testVertex.push_back(c);

	MarchingCube mc;
	mc.get_vertices_by_vtk("D:\\Runtime_SDK\\sources\\test_files\\SPH\\240201_SPHDambreak\\240201_SPHDambreak_grid1_1.vtk");
	//mc.get_vertices_by_txt("D:\\position2.txt", "D:\\density2.txt");
	//mc.make_polygon_with_particles(30.0);
	//mc.make_polygon_with_particles(testVertex, 20.0);
	//mc.print_txt("test.txt");
	//mc.print_vtu("test.vtu");
	//mc.generate_grid();
	return 0;
}