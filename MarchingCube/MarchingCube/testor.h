#pragma once
#include "data_structure.h"

class Testor {
private :
	std::vector<vec3> vertices;
	std::vector<Triangle> triangles;

public :
	Testor();
	~Testor();
	void setVertices(std::vector<vec3> vertices);
	bool verifyMC();
};