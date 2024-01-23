#include "MarchingCube.h"

bool MarchingCube::get_vertices_by_txt(std::string filepath)
{
	FILE* file = NULL;
	errno_t err;

	err = fopen_s(&file, filepath.c_str(), "rb");
	if (err != 0) {
		printf("failed to open File [%s]\n", filepath.c_str());
		return false;
	}

	
	fseek(file, 0, SEEK_END);
	long fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	printf("file size : %ld\n", fileSize);
	for (int i = 0; i < 100; ++i) {
		fread(&particles[i].position.x, sizeof(float), 1, file);
		fread(&particles[i].position.y, sizeof(float), 1, file);
		fread(&particles[i].position.z, sizeof(float), 1, file);
	}

	fclose(file);

	return true;
}

bool MarchingCube::make_polygon_with_particles(std::vector<vec3> vertices)
{
	for (int i = 0; i < vertices.size(); ++i)
	{
		//Particle tmpParticle = { vertices[i], 0.0 };
		particles.push_back(Particle{ vertices[i], 0.0 });
	}
	return true;
}

bool MarchingCube::make_polygon_with_particles()
{
	if (particles.size() == 0) {
		printf("[ERR] No particles\n");
		return false;
	}
}

bool MarchingCube::generate_grid()
{
	find_grid_minmax();

	vec3 tmpVertex = maxVertex - minVertex;
	gridSize = std::min(tmpVertex.x, std::min(tmpVertex.y, tmpVertex.z))/3;
	
	//int cellCnt = (int(tmpVertex.x/gridSize)+1) * (int(tmpVertex.y / gridSize)+1) * (int(tmpVertex.z / gridSize)+1);
	axisX = (int(tmpVertex.x / gridSize) + 1);
	axisY = (int(tmpVertex.y / gridSize) + 1);
	axisZ = (int(tmpVertex.z / gridSize) + 1);

	initialize_cell();
	
	printf("cells.x : %f\n", cells[0][1][2].coordinate.x);
	printf("cells.y : %f\n", cells[0][1][2].coordinate.y);
	printf("cells.z : %f\n", cells[0][1][2].coordinate.z);
	//cells = new Cell[2][3][4];
	//cells = new Cell[int(tmpVertex.x / gridSize) + 1][int(tmpVertex.y / gridSize) + 1][int(tmpVertex.z / gridSize) + 1];
	return true;
}

bool MarchingCube::put_density_into_cell()
{
	for (int i = 0; i < particles.size(); ++i)
	{
		cells[int(particles[i].position.x / gridSize)][int(particles[i].position.y / gridSize)][int(particles[i].position.y / gridSize)].particleCnt++;
	}

	for (int i = 0; i < particles.size(); ++i)
	{
		cells[int(particles[i].position.x / gridSize)][int(particles[i].position.y / gridSize)][int(particles[i].position.y / gridSize)].density += particles[i].density/ cells[int(particles[i].position.x / gridSize)][int(particles[i].position.y / gridSize)][int(particles[i].position.y / gridSize)].particleCnt;
	}

}

bool MarchingCube::initialize_cell()
{
	cells = new Cell * *[axisX];
	for (int i = 0; i < axisX; ++i)
	{
		cells[i] = new Cell * [axisY];
		for (int j = 0; j < axisY; ++j)
		{
			cells[i][j] = new Cell[axisZ];
		}
	}

	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				cells[i][j][k].coordinate = vec3{ 
					int(minVertex.x) + (gridSize / 2) + gridSize * i, 
					int(minVertex.y) + (gridSize / 2) + gridSize * j, 
					int(minVertex.z) + (gridSize / 2) + gridSize * k 
				};
				cells[i][j][k].set_vertex_with_coordinate(gridSize);
			}
		}
	}
	return true;
}

bool MarchingCube::find_grid_minmax()
{
	if (particles.size() == 0) {
		printf("[ERR] No particles\n");
		return false;
	}
	minVertex = particles[0].position;
	maxVertex = particles[0].position;

	for (int i = 0; i < particles.size(); ++i) {
		if (minVertex.x > particles[i].position.x) minVertex.x = particles[i].position.x;
		if (minVertex.y > particles[i].position.y) minVertex.y = particles[i].position.y;
		if (minVertex.z > particles[i].position.z) minVertex.z = particles[i].position.z;

		if (maxVertex.x < particles[i].position.x) maxVertex.x = particles[i].position.x;
		if (maxVertex.y < particles[i].position.y) maxVertex.y = particles[i].position.y;
		if (maxVertex.z < particles[i].position.z) maxVertex.z = particles[i].position.z;
	}
	return true;
}

void MarchingCube::print_txt(std::string filepath)
{
	FILE* file = NULL;

	fopen_s(&file, filepath.c_str(), "wb");

	for (int i = 0; i < triangles.size(); ++i) {
		fwrite(&triangles[i].t1.x, sizeof(float), 1, file);
		fwrite(&triangles[i].t1.y, sizeof(float), 1, file);
		fwrite(&triangles[i].t1.z, sizeof(float), 1, file);

		fwrite(&triangles[i].t2.x, sizeof(float), 1, file);
		fwrite(&triangles[i].t2.y, sizeof(float), 1, file);
		fwrite(&triangles[i].t2.z, sizeof(float), 1, file);

		fwrite(&triangles[i].t3.x, sizeof(float), 1, file);
		fwrite(&triangles[i].t3.y, sizeof(float), 1, file);
		fwrite(&triangles[i].t3.z, sizeof(float), 1, file);
	}

	fclose(file);
}