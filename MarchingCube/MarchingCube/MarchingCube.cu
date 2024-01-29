#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MarchingCube.cuh"

/*
struct vec3 {
	float x;
	float y;
	float z;

	__host__ __device__ vec3 operator*(float f) const { return vec3{ x * f, y * f, z * f }; }
	__host__ __device__ vec3 operator+(vec3 v1) const { return vec3{ x + v1.x, y + v1.y, z + v1.z }; }
	__host__ __device__ vec3 operator-(vec3 v1) const { return vec3{ x - v1.x, y - v1.y, z - v1.z }; }
};
*/

__global__ void compute_bit(Cell* cell, int x, int y, int z, float isoValue) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int zIndex = int(idx / ((x+1)*(y+1)));
	int yIndex = int(idx % ((x + 1) * (y + 1))) / (y+1);
	int xIndex = int(idx % ((x + 1) * (y + 1))) % (y + 1);
	//printf("bit %d\n", idx);
	if ((xIndex % x != 0) && (yIndex % y != 0) && (zIndex % z != 0))
	{
		float avgDensity = 0;
		avgDensity += cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex - 1)].density;
		avgDensity += cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex)].density;
		avgDensity += cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex)].density;
		avgDensity += cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex - 1)].density;
		avgDensity += cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex - 1)].density;
		avgDensity += cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex)].density;
		avgDensity += cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex)].density;
		avgDensity += cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex - 1)].density;
		//avgDensity *= 0.125;
		if (avgDensity > (isoValue * 8))
		{
			cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex - 1)].isUsingVertex[6] = true;
			cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex - 1)].valueOfVertex[6] = true;

			cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex)].isUsingVertex[7] = true;
			cell[(x * y * (zIndex - 1)) + (x * (yIndex - 1)) + (xIndex)].valueOfVertex[7] = true;

			cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex)].isUsingVertex[4] = true;
			cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex)].valueOfVertex[4] = true;

			cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex - 1)].isUsingVertex[5] = true;
			cell[(x * y * (zIndex - 1)) + (x * (yIndex)) + (xIndex - 1)].valueOfVertex[5] = true;

			cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex - 1)].isUsingVertex[2] = true;
			cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex - 1)].valueOfVertex[2] = true;

			cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex)].isUsingVertex[3] = true;
			cell[(x * y * (zIndex)) + (x * (yIndex - 1)) + (xIndex)].valueOfVertex[3] = true;

			cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex)].isUsingVertex[0] = true;
			cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex)].valueOfVertex[0] = true;

			cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex - 1)].isUsingVertex[1] = true;
			cell[(x * y * (zIndex)) + (x * (yIndex)) + (xIndex - 1)].valueOfVertex[1] = true;
		}
	}

	// point index (a,b,c) 일 때 
	// cell 기준 
	// cells 내 index			1차원 cell index					cell 내부의 vertex 번호
	// cell[a-1][b-1][c-1]		(x*y*(c-1)) + (x*(b-1)) + (a-1)		6
	// cell[a][b-1][c-1]		(x*y*(c-1)) + (x*(b-1)) + (a)		7
	// cell[a][b][c-1]			(x*y*(c-1)) + (x*(b)) + (a)			4
	// cell[a-1][b][c-1]		(x*y*(c-1)) + (x*(b)) + (a-1)		5
	// cell[a-1][b-1][c]		(x*y*(c)) + (x*(b-1)) + (a-1)		2
	// cell[a][b-1][c]			(x*y*(c)) + (x*(b-1)) + (a)			3
	// cell[a][b][c]			(x*y*(c)) + (x*(b)) + (a)			0
	// cell[a-1][b][c]			(x*y*(c)) + (x*(b)) + (a-1)			1

	//printf("xIndex : %d\tyIndex : %d\tzIndex : %d\n", xIndex, yIndex, zIndex);
	//printf("yIndex : %d\t", yIndex);
	//printf("test : %f\n", cell[idx].density);
	//printf("x : %f\ty : %f\tz : %f\n", cell[xIndex][yIndex][zIndex].coordinate.x, cell[xIndex][yIndex][zIndex].coordinate.y, cell[xIndex][yIndex][zIndex].coordinate.z);

	//printf("test : %d\n", idx);
}

/*
__global__ void compute_bit(Cell*** cell, int x, int y, int z) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int zIndex = int(idx % z);
	int yIndex = int((idx / z) % y);
	int xIndex = int(idx / (y * z));
	printf("xIndex : %d\tyIndex : %d\tzIndex : %d\n", xIndex, yIndex, zIndex);
	//printf("yIndex : %d\t", yIndex);
	printf("test : %d\n", cell[0][0][0].particleCnt);
	//printf("x : %f\ty : %f\tz : %f\n", cell[xIndex][yIndex][zIndex].coordinate.x, cell[xIndex][yIndex][zIndex].coordinate.y, cell[xIndex][yIndex][zIndex].coordinate.z);

	//printf("test : %d\n", idx);
}
*/

__global__ void make_cell_triangle(Cell* cell, int* d_edgeTable, short int* d_triTable, int x, int y, int z, float isoValue) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int usage = 0;
	//printf("cell x : %f\n", cell[idx].vertex[6].x);
	//printf("cell y : %f\n", cell[idx].vertex[6].y);
	//printf("cell z : %f\n", cell[idx].vertex[6].z);
	//printf("cell triangle %d\n", idx);
	if (cell[idx].isUsingVertex[0]) usage += 1;
	if (cell[idx].isUsingVertex[1]) usage += 2;
	if (cell[idx].isUsingVertex[2]) usage += 4;
	if (cell[idx].isUsingVertex[3]) usage += 8;
	if (cell[idx].isUsingVertex[4]) usage += 16;
	if (cell[idx].isUsingVertex[5]) usage += 32;
	if (cell[idx].isUsingVertex[6]) usage += 64;
	if (cell[idx].isUsingVertex[7]) usage += 128;

	int usingEdge = d_edgeTable[usage];

	if (usingEdge & 1) {
		cell[idx].edgeVertex[0] = cell[idx].vertex[0] + ((cell[idx].vertex[1] - cell[idx].vertex[0]) * ((isoValue - cell[idx].valueOfVertex[0]) / (cell[idx].valueOfVertex[1] - cell[idx].valueOfVertex[0])));

		//printf("triangleArr X : %f\n", cell[idx].vertex[1].x);
		//printf("triangleArr Y : %f\n", cell[idx].vertex[1].y);
		//printf("triangleArr Z : %f\n", cell[idx].vertex[1].z);
	}
	if (usingEdge & 2)		cell[idx].edgeVertex[1] = cell[idx].vertex[1] + ((cell[idx].vertex[2] - cell[idx].vertex[1]) * ((isoValue - cell[idx].valueOfVertex[1]) / (cell[idx].valueOfVertex[2] - cell[idx].valueOfVertex[1])));
	if (usingEdge & 4)		cell[idx].edgeVertex[2] = cell[idx].vertex[2] + ((cell[idx].vertex[3] - cell[idx].vertex[2]) * ((isoValue - cell[idx].valueOfVertex[2]) / (cell[idx].valueOfVertex[3] - cell[idx].valueOfVertex[2])));
	if (usingEdge & 8)		cell[idx].edgeVertex[3] = cell[idx].vertex[3] + ((cell[idx].vertex[0] - cell[idx].vertex[3]) * ((isoValue - cell[idx].valueOfVertex[3]) / (cell[idx].valueOfVertex[4] - cell[idx].valueOfVertex[3])));

	if (usingEdge & 16)		cell[idx].edgeVertex[4] = cell[idx].vertex[4] + ((cell[idx].vertex[5] - cell[idx].vertex[4]) * ((isoValue - cell[idx].valueOfVertex[4]) / (cell[idx].valueOfVertex[5] - cell[idx].valueOfVertex[4])));
	if (usingEdge & 32)		cell[idx].edgeVertex[5] = cell[idx].vertex[5] + ((cell[idx].vertex[6] - cell[idx].vertex[5]) * ((isoValue - cell[idx].valueOfVertex[5]) / (cell[idx].valueOfVertex[6] - cell[idx].valueOfVertex[5])));
	if (usingEdge & 64)		cell[idx].edgeVertex[6] = cell[idx].vertex[6] + ((cell[idx].vertex[7] - cell[idx].vertex[6]) * ((isoValue - cell[idx].valueOfVertex[6]) / (cell[idx].valueOfVertex[7] - cell[idx].valueOfVertex[6])));
	if (usingEdge & 128)	cell[idx].edgeVertex[7] = cell[idx].vertex[7] + ((cell[idx].vertex[4] - cell[idx].vertex[7]) * ((isoValue - cell[idx].valueOfVertex[7]) / (cell[idx].valueOfVertex[4] - cell[idx].valueOfVertex[7])));
	
	if (usingEdge & 256)	cell[idx].edgeVertex[8] = cell[idx].vertex[0] + ((cell[idx].vertex[4] - cell[idx].vertex[0]) * ((isoValue - cell[idx].valueOfVertex[0]) / (cell[idx].valueOfVertex[4] - cell[idx].valueOfVertex[0])));
	if (usingEdge & 512)	cell[idx].edgeVertex[9] = cell[idx].vertex[1] + ((cell[idx].vertex[5] - cell[idx].vertex[1]) * ((isoValue - cell[idx].valueOfVertex[1]) / (cell[idx].valueOfVertex[5] - cell[idx].valueOfVertex[1])));
	if (usingEdge & 1024)	cell[idx].edgeVertex[10] = cell[idx].vertex[2] + ((cell[idx].vertex[6] - cell[idx].vertex[2]) * ((isoValue - cell[idx].valueOfVertex[2]) / (cell[idx].valueOfVertex[6] - cell[idx].valueOfVertex[2])));
	if (usingEdge & 2048)	cell[idx].edgeVertex[11] = cell[idx].vertex[3] + ((cell[idx].vertex[7] - cell[idx].vertex[3]) * ((isoValue - cell[idx].valueOfVertex[3]) / (cell[idx].valueOfVertex[7] - cell[idx].valueOfVertex[3])));
	/*
	printf("triangleArr X : %f\n", cell[idx].edgeVertex[0].x);
	printf("triangleArr Y : %f\n", cell[idx].edgeVertex[0].y);
	printf("triangleArr Z : %f\n", cell[idx].edgeVertex[0].z);
	*/
	for (int i = 0; i < 5; i++)
	{
		if (d_triTable[(usage * 16) + (i * 3)] == -1) {
			cell[idx].triangleCnt = i;
			break;
		}
		cell[idx].triangles[i].t1 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i*3)]];
		cell[idx].triangles[i].t2 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i*3)+1]];
		cell[idx].triangles[i].t3 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i*3)+2]];
		//printf("%d, %d, %d\n", (usage * 16) + (i * 3), (usage * 16) + (i * 3) + 1, (usage * 16) + (i * 3) + 2);
		//printf("(%f, %f, %f)\n", cell[idx].triangles[i].t1.x, cell[idx].triangles[i].t2.x, cell[idx].triangles[i].t3.x);
	}
	/*
	printf("triangleArr X : %f\n", cell[idx].triangles[0].t1.x);
	printf("triangleArr Y : %f\n", cell[idx].triangles[0].t1.y);
	printf("triangleArr Z : %f\n", cell[idx].triangles[0].t1.z);
	*/
}

__global__ void add_triangle_to_array(Cell* cell, Triangle* triangleArr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int amount = cell[idx].triangleCnt;
	triangleArr[idx * 5 + 0] = cell[idx].triangles[0];
	triangleArr[idx * 5 + 1] = cell[idx].triangles[1];
	triangleArr[idx * 5 + 2] = cell[idx].triangles[2];
	triangleArr[idx * 5 + 3] = cell[idx].triangles[3];
	triangleArr[idx * 5 + 4] = cell[idx].triangles[4];

	//printf("(%f, %f, %f)\n", cell[idx].triangles[0].t1.x, cell[idx].triangles[0].t1.y, cell[idx].triangles[0].t1.z);
}
__global__ void compute_triangle_index(Cell* cell) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//cell[idx].triangleStartIndex
}
void MarchingCube::compute_cell_bit(float isoValue)
{
	compute_bit << <1, (axisX+1)*(axisY+1)*(axisZ+1) >> > (d_cells, axisX, axisY, axisZ, isoValue);
}

void MarchingCube::make_triangle(float isoValue)
{
	make_cell_triangle <<< 1, axisX* axisY* axisZ >> > (d_cells, d_edgeTable, d_triangleTable, axisX, axisY, axisZ, isoValue);
}

void MarchingCube::make_triangle_arr()
{
	//add_triangle_to_array <<< 1, (axisX)* (axisY)* (axisZ) >> > (d_cells, d_triangles);
	//cudaMemcpy(cells, d_triangles, axisX * axisY * axisZ * sizeof(Triangle), cudaMemcpyDeviceToHost);
	int d_idx = 0;
	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				cudaMemcpy(&cells[i][j][k], &d_cells[d_idx], sizeof(Cell), cudaMemcpyDeviceToHost);
				d_idx++;
			}
		}
	}

	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				for (int l = 0; l < cells[i][j][k].triangleCnt; ++l)
				{
					h_triangles.push_back(cells[i][j][k].triangles[l]);
				}
			}
		}
	}

	//	for (int i = 0 ; i < axisX * axisY * axisZ; ++i)
	//{
		
	//}
	//cudaMemcpy(h_triangles, d_triangles, axisX * axisY * axisZ * sizeof(Triangle), cudaMemcpyDeviceToHost);
}

__global__ void check_vertex(Cell* cell) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	printf("cell x : %f\n", cell[idx].vertex[6].x);
	printf("cell y : %f\n", cell[idx].vertex[6].y);
	printf("cell z : %f\n", cell[idx].vertex[6].z);
	//printf("idx : %d\n", idx);
}

void MarchingCube::alloc_device_memory()
{
	int d_idx = 0;
	cudaMalloc((void**)&d_cells, axisX * axisY * axisZ * sizeof(Cell));
	cudaMalloc((void**)&d_edgeTable, 256 * sizeof(int));
	cudaMalloc((void**)&d_triangleTable, 256 * 16 * sizeof(short int));
	cudaMalloc((void**)&d_triangles, axisX * axisY * axisZ * sizeof(Triangle));

	cudaMemcpy(d_edgeTable, EdgeTable, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangleTable, TriTable, 256 * 16 * sizeof(short int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_triangles, h_triangles, axisX * axisY * axisZ * sizeof(Triangle), cudaMemcpyHostToDevice);

	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				//printf("density : %f\n", cells[i][j][k].density);
				cells[i][j][k].index = d_idx;
				cudaMemcpy(&d_cells[d_idx], &cells[i][j][k], sizeof(Cell), cudaMemcpyHostToDevice);
				d_idx++;
			}
		}
	}

	//check_vertex << < 1, (axisX)* (axisY)* (axisZ) >> > (d_cells);
}

void MarchingCube::free_device_memory()
{
	cudaFree(d_cells);
	cudaFree(d_edgeTable);
	cudaFree(d_triangleTable);
	//cudaFree(d_triangles);
}

bool MarchingCube::get_vertices_by_txt(std::string positionPath, std::string densityPath)
{
	FILE* file = NULL;
	errno_t err;

	err = fopen_s(&file, positionPath.c_str(), "rb");
	if (err != 0) {
		printf("failed to open File [%s]\n", positionPath.c_str());
		return false;
	}

	fseek(file, 0, SEEK_END);
	long fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	printf("file size : %ld\n", fileSize/(3*sizeof(float)));

	particles = new Particle[fileSize / (3 * sizeof(float))];

	particleSize = fileSize / (3 * sizeof(float));
	for (int i = 0; i < fileSize/(3*sizeof(float)); ++i) {
		vec3 tmpPosition;
		fread(&(particles[i].position.x), sizeof(float), 1, file);
		fread(&(particles[i].position.y), sizeof(float), 1, file);
		fread(&(particles[i].position.z), sizeof(float), 1, file);

		//Particle{ tmpPosition}
		//printf("%f\n", particles[i].position.x);
	}

	fclose(file);

	//printf("testsetset1111\n");


	FILE* file2 = NULL;
	errno_t err2;
	err2 = fopen_s(&file2, densityPath.c_str(), "rb");
	if (err2 != 0) {
		printf("failed to open File [%s]\n", densityPath.c_str());
		return false;
	}

	fseek(file2, 0, SEEK_END);
	long fileSize2 = ftell(file2);
	fseek(file2, 0, SEEK_SET);

	printf("file size : %ld\n", fileSize / (3 * sizeof(float)));

	for (int i = 0; i < fileSize / (3 * sizeof(float)); ++i) {
		fread(&(particles[i].density), sizeof(float), 1, file2);
	}

	fclose(file2);

	return true;
}

bool MarchingCube::make_polygon_with_particles(std::vector<vec3> vertices, float isoValue)
{
	particles = new Particle[vertices.size()];
	particleSize = vertices.size();

	for (int i = 0; i < vertices.size(); ++i)
	{
		//Particle tmpParticle = { vertices[i], 0.0 };
		particles[i].position = vertices[i];
		particles[i].density = 94.0;
	}
	generate_grid();

	printf("x Size : %d\ty Size : %d\tz Size : %d\n", axisX, axisY, axisZ);

	if (put_density_into_cell()) printf("put density into cell \n");

	alloc_device_memory();

	compute_cell_bit(isoValue);

	make_triangle(isoValue);

	make_triangle_arr();

	return true;
}

bool MarchingCube::make_polygon_with_particles(float isoValue)
{
	if (particleSize == 0) {
		printf("[ERR] No particles\n");
		return false;
	}
	generate_grid();

	printf("x Size : %d\ty Size : %d\tz Size : %d\n", axisX, axisY, axisZ);

	if (put_density_into_cell()) printf("put density into cell \n");

	alloc_device_memory();

	compute_cell_bit(isoValue);

	make_triangle(isoValue);

	make_triangle_arr();

	return true;
}

bool MarchingCube::generate_grid()
{
	find_grid_minmax();

	vec3 tmpVertex = maxVertex - minVertex;
	gridSize = std::min(tmpVertex.x, std::min(tmpVertex.y, tmpVertex.z)) / 6;

	//int cellCnt = (int(tmpVertex.x/gridSize)+1) * (int(tmpVertex.y / gridSize)+1) * (int(tmpVertex.z / gridSize)+1);
	axisX = (int(tmpVertex.x / gridSize) + 1);
	axisY = (int(tmpVertex.y / gridSize) + 1);
	axisZ = (int(tmpVertex.z / gridSize) + 1);

	printf("%d, %d, %d\n", axisX, axisY, axisZ);
	initialize_cell();

	//cells = new Cell[2][3][4];
	//cells = new Cell[int(tmpVertex.x / gridSize) + 1][int(tmpVertex.y / gridSize) + 1][int(tmpVertex.z / gridSize) + 1];
	return true;
}

bool MarchingCube::put_density_into_cell()
{
	for (int i = 0; i < particleSize; ++i)
	{
		cells[int((particles[i].position.x - minVertex.x) / gridSize)][int((particles[i].position.y - minVertex.y) / gridSize)][int((particles[i].position.z - minVertex.z) / gridSize)].particleCnt++;
	}

	for (int i = 0; i < particleSize; ++i)
	{
		cells[int((particles[i].position.x - minVertex.x) / gridSize)][int((particles[i].position.y - minVertex.y) / gridSize)][int((particles[i].position.z - minVertex.z) / gridSize)].density += particles[i].density / cells[int((particles[i].position.x - minVertex.x) / gridSize)][int((particles[i].position.y - minVertex.y) / gridSize)][int((particles[i].position.z - minVertex.z) / gridSize)].particleCnt;
	}
	return true;
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

	//h_triangles = new Triangle[axisX * axisY * axisZ];

	return true;
}

bool MarchingCube::find_grid_minmax()
{
	if (particleSize == 0) {
		printf("[ERR] No particles\n");
		return false;
	}
	minVertex = particles[0].position;
	maxVertex = particles[0].position;

	for (int i = 0; i < particleSize; ++i) {
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

	for (int i = 0; i < h_triangles.size(); ++i) {
		printf("triangle %d\n", i);

		fwrite(&h_triangles[i].t1.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t1.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t1.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t1.x, h_triangles[i].t1.y, h_triangles[i].t1.z);

		fwrite(&h_triangles[i].t2.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t2.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t2.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t2.x, h_triangles[i].t2.y, h_triangles[i].t2.z);

		fwrite(&h_triangles[i].t3.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t3.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t3.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t3.x, h_triangles[i].t3.y, h_triangles[i].t3.z);
	}

	fclose(file);
}

void MarchingCube::print_vtu(std::string filepath)
{
	FILE* file = NULL;

	fopen_s(&file, filepath.c_str(), "wb");

	std::string txt = "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<UnstructuredGrid>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "<Piece NumberOfPoints=\"" + std::to_string(3) + "\" NumberOfCells=\"" + std::to_string(1) + "\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "<Points>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "<DataArray type=\"Float64\" NumberOfComponents=\"" + std::to_string(h_triangles.size()*3) + "\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	for (int i = 0; i < h_triangles.size(); ++i) {
		printf("triangle %d\n", i);


		fwrite(&h_triangles[i].t1.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t1.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t1.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t1.x, h_triangles[i].t1.y, h_triangles[i].t1.z);

		fwrite(&h_triangles[i].t2.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t2.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t2.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t2.x, h_triangles[i].t2.y, h_triangles[i].t2.z);

		fwrite(&h_triangles[i].t3.x, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t3.y, sizeof(float), 1, file);
		fwrite(&h_triangles[i].t3.z, sizeof(float), 1, file);
		printf("(%f, %f, %f)\n", h_triangles[i].t3.x, h_triangles[i].t3.y, h_triangles[i].t3.z);
	}

	txt = "\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</Points>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<Cells>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = std::to_string(h_triangles.size() * 3) + "\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "";
	for (int i = 0; i < h_triangles.size()*3; ++i)
	{
		if (i % 3 == 2) txt += (std::to_string(i) + "\n");
		else txt += (std::to_string(i) + " ");
	}
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "5\n</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	fclose(file);
}

void MarchingCube::write_binary(std::string txt)
{

}