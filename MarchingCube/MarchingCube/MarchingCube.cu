#include <cuda_runtime.h>
#include "MarchingCube.cuh"

__global__ void compute_bit(Cell*** cell) {
	printf("test\n");
}

void compute_cell_bit(Cell**** cells)
{
	Cell*** d_cells;

	cudaMalloc(&d_cells, sizeof(cells));

	cudaMemcpy(d_cells, cells, sizeof(cells), cudaMemcpyHostToDevice);

	compute_bit << <1, 15 >> > (d_cells);

	cudaFree(d_cells);
}

void free_gpu_memory()
{

}