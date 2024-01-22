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

	printf("file size : %ld", fileSize);
	for (int i = 0; i < 100; ++i) {
		fread(&vertices[i].x, sizeof(float), 1, file);
		fread(&vertices[i].y, sizeof(float), 1, file);
		fread(&vertices[i].z, sizeof(float), 1, file);
	}

	fclose(file);

	return true;
}

void MarchingCube::make_marchingCube_with_vertices(std::vector<vec3> vertices)
{
	
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