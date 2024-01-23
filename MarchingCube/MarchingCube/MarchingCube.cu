﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MarchingCube.cuh"

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

void compute_cell_bit(Cell*** cells, int axisX, int axisY, int axisZ)
{
	Cell*** d_cells;

	cudaMalloc(&d_cells, axisX * axisY * axisZ* sizeof(Cell));

	cudaMemcpy(d_cells, cells, axisX * axisY * axisZ * sizeof(Cell), cudaMemcpyDeviceToDevice);
	printf("test : %d\n", cells[0][0][0].particleCnt);
	compute_bit << <1, axisX*axisY*axisZ >> > (d_cells, axisX, axisY, axisZ);

	cudaFree(d_cells);
}