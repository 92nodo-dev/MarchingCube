#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MarchingCube.cuh"

__global__ void compute_bit(Cell* cell, int x, int y, int z) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int zIndex = int(idx / ((x+1)*(y+1)));
	int yIndex = int(idx % ((x + 1) * (y + 1))) / (y+1);
	int xIndex = int(idx % ((x + 1) * (y + 1))) % (y + 1);

	// point index (a,b,c) 일 때 
	// cell 기준 
	// cells 내 index			1차원 cell index					cell 내부의 vertex 번호
	// cell[a-1][b-1][c-1]		(x*y*(c-1)) + (x*(b-1)) + (a-1)		6
	// cell[a][b-1][c-1]		(x*y*(c-1)) + (x*(b-1)) + (a-1)		7
	// cell[a][b][c-1]			(x*y*(c-1)) + (x*(b-1)) + (a-1)		4
	// cell[a-1][b][c-1]		(x*y*(c-1)) + (x*(b-1)) + (a-1)		5
	// cell[a-1][b-1][c]		(x*y*(c-1)) + (x*(b-1)) + (a-1)		2
	// cell[a][b-1][c]			(x*y*(c-1)) + (x*(b-1)) + (a-1)		3
	// cell[a][b][c]			(x*y*(c-1)) + (x*(b-1)) + (a-1)		0
	// cell[a-1][b][c]			(x*y*(c-1)) + (x*(b-1)) + (a-1)		1

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



void compute_cell_bit(Cell*** cells, int axisX, int axisY, int axisZ)
{
	//Cell*** d_cells;
	Cell* d_cells;
	int d_idx = 0;
	cudaMalloc((void**)&d_cells, axisX * axisY * axisZ* sizeof(Cell));

	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				//printf("density : %f\n", cells[i][j][k].density);
				cudaMemcpy(&d_cells[d_idx], &cells[i][j][k], sizeof(Cell), cudaMemcpyHostToDevice);
				d_idx++;
			}
		}
	}
	//cudaMemcpy(d_cells, cells, axisX * axisY * axisZ * sizeof(Cell), cudaMemcpyDeviceToDevice);
	
	compute_bit << <1, (axisX+1)*(axisY+1)*(axisZ+1) >> > (d_cells, axisX, axisY, axisZ);

	cudaFree(d_cells);
}