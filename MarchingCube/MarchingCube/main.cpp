#include "testor.h"
#include "MarchingCube.h"
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
	mc.make_polygon_with_particles(testVertex);
	mc.make_grid();
	return 0;
}