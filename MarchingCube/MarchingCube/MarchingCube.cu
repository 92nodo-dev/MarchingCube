#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MarchingCube.cuh"

__global__ void compute_bit(Cell* cell, int x, int y, int z, float isoValue) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int xIndex = int(idx / ((z + 1) * (y + 1)));
	int yIndex = int(idx % ((z + 1) * (y + 1))) / (z + 1);
	int zIndex = int(idx % ((z + 1) * (y + 1))) % (z + 1);

	if ((xIndex % x != 0) && (yIndex % y != 0) && (zIndex % z != 0))
	{
		float avgDensity = 0;

		avgDensity += cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex - 1)].density;
		avgDensity += cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex)].density;
		avgDensity += cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex)].density;
		avgDensity += cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex - 1)].density;
		avgDensity += cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex - 1)].density;
		avgDensity += cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex)].density;
		avgDensity += cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex)].density;
		avgDensity += cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex - 1)].density;

		avgDensity *= 0.125;

		cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex - 1)].valueOfVertex[6] = avgDensity;
		cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex)].valueOfVertex[2] = avgDensity;
		cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex)].valueOfVertex[1] = avgDensity;
		cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex - 1)].valueOfVertex[5] = avgDensity;
		cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex - 1)].valueOfVertex[7] = avgDensity;
		cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex)].valueOfVertex[3] = avgDensity;
		cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex)].valueOfVertex[0] = avgDensity;
		cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex - 1)].valueOfVertex[4] = avgDensity;

		if (avgDensity < isoValue)
		{
			//printf("idx : %d\n", idx);
			cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex - 1)].isUsingVertex[6] = true;
			cell[(z * y * (xIndex - 1)) + (z * (yIndex - 1)) + (zIndex)].isUsingVertex[2] = true;
			cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex)].isUsingVertex[1] = true;
			cell[(z * y * (xIndex - 1)) + (z * (yIndex)) + (zIndex - 1)].isUsingVertex[5] = true;
			cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex - 1)].isUsingVertex[7] = true;
			cell[(z * y * (xIndex)) + (z * (yIndex - 1)) + (zIndex)].isUsingVertex[3] = true;
			cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex)].isUsingVertex[0] = true;
			cell[(z * y * (xIndex)) + (z * (yIndex)) + (zIndex - 1)].isUsingVertex[4] = true;
		}
	}
}

__global__ void make_cell_triangle(Cell* cell, int* d_edgeTable, short int* d_triTable, int x, int y, int z, float isoValue) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int usage = 0;

	if (cell[idx].isUsingVertex[0]) usage += 1;
	if (cell[idx].isUsingVertex[1]) usage += 2;
	if (cell[idx].isUsingVertex[2]) usage += 4;
	if (cell[idx].isUsingVertex[3]) usage += 8;
	if (cell[idx].isUsingVertex[4]) usage += 16;
	if (cell[idx].isUsingVertex[5]) usage += 32;
	if (cell[idx].isUsingVertex[6]) usage += 64;
	if (cell[idx].isUsingVertex[7]) usage += 128;

	int usingEdge = d_edgeTable[usage];

	// max = v2, min = v1
	
	float k1, k2, v1, v2;
	vec3 p1, p2;
	if (usingEdge & 1)
	{
		v1 = cell[idx].valueOfVertex[1] < cell[idx].valueOfVertex[0] ? cell[idx].valueOfVertex[1] : cell[idx].valueOfVertex[0];
		v2 = cell[idx].valueOfVertex[1] > cell[idx].valueOfVertex[0] ? cell[idx].valueOfVertex[1] : cell[idx].valueOfVertex[0];
		p1 = cell[idx].valueOfVertex[1] < cell[idx].valueOfVertex[0] ? cell[idx].vertex[1] : cell[idx].vertex[0];
		p2 = cell[idx].valueOfVertex[1] > cell[idx].valueOfVertex[0] ? cell[idx].vertex[1] : cell[idx].vertex[0];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[0] = (cell[idx].vertex[0] + cell[idx].vertex[1]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[0] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 2)
	{
		v1 = cell[idx].valueOfVertex[2] < cell[idx].valueOfVertex[1] ? cell[idx].valueOfVertex[2] : cell[idx].valueOfVertex[1];
		v2 = cell[idx].valueOfVertex[2] > cell[idx].valueOfVertex[1] ? cell[idx].valueOfVertex[2] : cell[idx].valueOfVertex[1];
		p1 = cell[idx].valueOfVertex[2] < cell[idx].valueOfVertex[1] ? cell[idx].vertex[2] : cell[idx].vertex[1];
		p2 = cell[idx].valueOfVertex[2] > cell[idx].valueOfVertex[1] ? cell[idx].vertex[2] : cell[idx].vertex[1];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[1] = (cell[idx].vertex[1] + cell[idx].vertex[2]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[1] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 4)
	{
		v1 = cell[idx].valueOfVertex[3] < cell[idx].valueOfVertex[2] ? cell[idx].valueOfVertex[3] : cell[idx].valueOfVertex[2];
		v2 = cell[idx].valueOfVertex[3] > cell[idx].valueOfVertex[2] ? cell[idx].valueOfVertex[3] : cell[idx].valueOfVertex[2];
		p1 = cell[idx].valueOfVertex[3] < cell[idx].valueOfVertex[2] ? cell[idx].vertex[3] : cell[idx].vertex[2];
		p2 = cell[idx].valueOfVertex[3] > cell[idx].valueOfVertex[2] ? cell[idx].vertex[3] : cell[idx].vertex[2];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[2] = (cell[idx].vertex[3] + cell[idx].vertex[2]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[2] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 8)
	{
		v1 = cell[idx].valueOfVertex[0] < cell[idx].valueOfVertex[3] ? cell[idx].valueOfVertex[0] : cell[idx].valueOfVertex[3];
		v2 = cell[idx].valueOfVertex[0] > cell[idx].valueOfVertex[3] ? cell[idx].valueOfVertex[0] : cell[idx].valueOfVertex[3];
		p1 = cell[idx].valueOfVertex[0] < cell[idx].valueOfVertex[3] ? cell[idx].vertex[0] : cell[idx].vertex[3];
		p2 = cell[idx].valueOfVertex[0] > cell[idx].valueOfVertex[3] ? cell[idx].vertex[0] : cell[idx].vertex[3];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[3] = (cell[idx].vertex[0] + cell[idx].vertex[3]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[3] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 16)
	{
		v1 = cell[idx].valueOfVertex[4] < cell[idx].valueOfVertex[5] ? cell[idx].valueOfVertex[4] : cell[idx].valueOfVertex[5];
		v2 = cell[idx].valueOfVertex[4] > cell[idx].valueOfVertex[5] ? cell[idx].valueOfVertex[4] : cell[idx].valueOfVertex[5];
		p1 = cell[idx].valueOfVertex[4] < cell[idx].valueOfVertex[5] ? cell[idx].vertex[4] : cell[idx].vertex[5];
		p2 = cell[idx].valueOfVertex[4] > cell[idx].valueOfVertex[5] ? cell[idx].vertex[4] : cell[idx].vertex[5];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[4] = (cell[idx].vertex[4] + cell[idx].vertex[5]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[4] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 32)
	{
		v1 = cell[idx].valueOfVertex[5] < cell[idx].valueOfVertex[6] ? cell[idx].valueOfVertex[5] : cell[idx].valueOfVertex[6];
		v2 = cell[idx].valueOfVertex[5] > cell[idx].valueOfVertex[6] ? cell[idx].valueOfVertex[5] : cell[idx].valueOfVertex[6];
		p1 = cell[idx].valueOfVertex[5] < cell[idx].valueOfVertex[6] ? cell[idx].vertex[5] : cell[idx].vertex[6];
		p2 = cell[idx].valueOfVertex[5] > cell[idx].valueOfVertex[6] ? cell[idx].vertex[5] : cell[idx].vertex[6];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[5] = (cell[idx].vertex[5] + cell[idx].vertex[6]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[5] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 64)
	{
		v1 = cell[idx].valueOfVertex[6] < cell[idx].valueOfVertex[7] ? cell[idx].valueOfVertex[6] : cell[idx].valueOfVertex[7];
		v2 = cell[idx].valueOfVertex[6] > cell[idx].valueOfVertex[7] ? cell[idx].valueOfVertex[6] : cell[idx].valueOfVertex[7];
		p1 = cell[idx].valueOfVertex[6] < cell[idx].valueOfVertex[7] ? cell[idx].vertex[6] : cell[idx].vertex[7];
		p2 = cell[idx].valueOfVertex[6] > cell[idx].valueOfVertex[7] ? cell[idx].vertex[6] : cell[idx].vertex[7];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[6] = (cell[idx].vertex[6] + cell[idx].vertex[7]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[6] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 128)
	{
		v1 = cell[idx].valueOfVertex[7] < cell[idx].valueOfVertex[4] ? cell[idx].valueOfVertex[7] : cell[idx].valueOfVertex[4];
		v2 = cell[idx].valueOfVertex[7] > cell[idx].valueOfVertex[4] ? cell[idx].valueOfVertex[7] : cell[idx].valueOfVertex[4];
		p1 = cell[idx].valueOfVertex[7] < cell[idx].valueOfVertex[4] ? cell[idx].vertex[7] : cell[idx].vertex[4];
		p2 = cell[idx].valueOfVertex[7] > cell[idx].valueOfVertex[4] ? cell[idx].vertex[7] : cell[idx].vertex[4];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[7] = (cell[idx].vertex[7] + cell[idx].vertex[4]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[7] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 256)
	{
		v1 = cell[idx].valueOfVertex[0] < cell[idx].valueOfVertex[4] ? cell[idx].valueOfVertex[0] : cell[idx].valueOfVertex[4];
		v2 = cell[idx].valueOfVertex[0] > cell[idx].valueOfVertex[4] ? cell[idx].valueOfVertex[0] : cell[idx].valueOfVertex[4];
		p1 = cell[idx].valueOfVertex[0] < cell[idx].valueOfVertex[4] ? cell[idx].vertex[0] : cell[idx].vertex[4];
		p2 = cell[idx].valueOfVertex[0] > cell[idx].valueOfVertex[4] ? cell[idx].vertex[0] : cell[idx].vertex[4];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[8] = (cell[idx].vertex[0] + cell[idx].vertex[4]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[8] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 512)
	{
		v1 = cell[idx].valueOfVertex[1] < cell[idx].valueOfVertex[5] ? cell[idx].valueOfVertex[1] : cell[idx].valueOfVertex[5];
		v2 = cell[idx].valueOfVertex[1] > cell[idx].valueOfVertex[5] ? cell[idx].valueOfVertex[1] : cell[idx].valueOfVertex[5];
		p1 = cell[idx].valueOfVertex[1] < cell[idx].valueOfVertex[5] ? cell[idx].vertex[1] : cell[idx].vertex[5];
		p2 = cell[idx].valueOfVertex[1] > cell[idx].valueOfVertex[5] ? cell[idx].vertex[1] : cell[idx].vertex[5];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[9] = (cell[idx].vertex[1] + cell[idx].vertex[5]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[9] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 1024)
	{
		v1 = cell[idx].valueOfVertex[2] < cell[idx].valueOfVertex[6] ? cell[idx].valueOfVertex[2] : cell[idx].valueOfVertex[6];
		v2 = cell[idx].valueOfVertex[2] > cell[idx].valueOfVertex[6] ? cell[idx].valueOfVertex[2] : cell[idx].valueOfVertex[6];
		p1 = cell[idx].valueOfVertex[2] < cell[idx].valueOfVertex[6] ? cell[idx].vertex[2] : cell[idx].vertex[6];
		p2 = cell[idx].valueOfVertex[2] > cell[idx].valueOfVertex[6] ? cell[idx].vertex[2] : cell[idx].vertex[6];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[10] = (cell[idx].vertex[2] + cell[idx].vertex[6]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[10] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}
	if (usingEdge & 2048)
	{
		v1 = cell[idx].valueOfVertex[3] < cell[idx].valueOfVertex[7] ? cell[idx].valueOfVertex[3] : cell[idx].valueOfVertex[7];
		v2 = cell[idx].valueOfVertex[3] > cell[idx].valueOfVertex[7] ? cell[idx].valueOfVertex[3] : cell[idx].valueOfVertex[7];
		p1 = cell[idx].valueOfVertex[3] < cell[idx].valueOfVertex[7] ? cell[idx].vertex[3] : cell[idx].vertex[7];
		p2 = cell[idx].valueOfVertex[3] > cell[idx].valueOfVertex[7] ? cell[idx].vertex[3] : cell[idx].vertex[7];
		k1 = v2 - isoValue;
		k2 = isoValue - v1;
		if ((k1 <= 0) || (k2 <= 0)) {
			cell[idx].edgeVertex[11] = (cell[idx].vertex[3] + cell[idx].vertex[7]) * 0.5f;
			return;
		}
		cell[idx].edgeVertex[11] = ((p1 * k1) + (p2 * k2)) * (1 / (k1 + k2));
	}

/*
	if (usingEdge & 1)		cell[idx].edgeVertex[0] = cell[idx].vertex[0] +((cell[idx].vertex[1] - cell[idx].vertex[0]) * ((isoValue - cell[idx].valueOfVertex[0]) / (cell[idx].valueOfVertex[1] - cell[idx].valueOfVertex[0])));
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
*/

/*
	if (usingEdge & 1)		cell[idx].edgeVertex[0] = (cell[idx].vertex[0] + cell[idx].vertex[1]) * 0.5f;
	if (usingEdge & 2)		cell[idx].edgeVertex[1] = (cell[idx].vertex[1] + cell[idx].vertex[2]) * 0.5f;
	if (usingEdge & 4)		cell[idx].edgeVertex[2] = (cell[idx].vertex[2] + cell[idx].vertex[3]) * 0.5f; 
	if (usingEdge & 8)		cell[idx].edgeVertex[3] = (cell[idx].vertex[0] + cell[idx].vertex[3]) * 0.5f;

	if (usingEdge & 16)		cell[idx].edgeVertex[4] = (cell[idx].vertex[4] + cell[idx].vertex[5]) * 0.5f;
	if (usingEdge & 32)		cell[idx].edgeVertex[5] = (cell[idx].vertex[5] + cell[idx].vertex[6]) * 0.5f; 
	if (usingEdge & 64)		cell[idx].edgeVertex[6] = (cell[idx].vertex[6] + cell[idx].vertex[7]) * 0.5f;
	if (usingEdge & 128)	cell[idx].edgeVertex[7] = (cell[idx].vertex[7] + cell[idx].vertex[4]) * 0.5f;

	if (usingEdge & 256)	cell[idx].edgeVertex[8] = (cell[idx].vertex[0] + cell[idx].vertex[4]) * 0.5f; 
	if (usingEdge & 512)	cell[idx].edgeVertex[9] = (cell[idx].vertex[1] + cell[idx].vertex[5]) * 0.5f; 
	if (usingEdge & 1024)	cell[idx].edgeVertex[10] = (cell[idx].vertex[2] + cell[idx].vertex[6]) * 0.5f; 
	if (usingEdge & 2048)	cell[idx].edgeVertex[11] = (cell[idx].vertex[3] + cell[idx].vertex[7]) * 0.5f; 
*/

	for (int i = 0; i < 5; i++)
	{
		if (d_triTable[(usage * 16) + (i * 3)] == -1) {
			cell[idx].triangleCnt = i;
			break;
		}
		cell[idx].triangles[i].t1 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i * 3)]];
		cell[idx].triangles[i].t2 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i * 3) + 1]];
		cell[idx].triangles[i].t3 = cell[idx].edgeVertex[d_triTable[(usage * 16) + (i * 3) + 2]];

		if (d_triTable[(usage * 16) + (i * 3)] == 0) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[1]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 1) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[2]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 2) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[3]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 3) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[0]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 4) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[4] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 5) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[5] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 6) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[6] + cell[idx].valueOfVertex[7]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 7) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[7] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 8) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 9) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 10) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3)] == 11) cell[idx].triangles[i].density[0] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[7]) * 0.5f;

		if (d_triTable[(usage * 16) + (i * 3) + 1] == 0) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[1]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 1) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[2]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 2) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[3]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 3) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[0]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 4) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[4] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 5) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[5] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 6) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[6] + cell[idx].valueOfVertex[7]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 7) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[7] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 8) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 9) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 10) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 1] == 11) cell[idx].triangles[i].density[1] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[7]) * 0.5f;

		if (d_triTable[(usage * 16) + (i * 3) + 2] == 0) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[1]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 1) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[2]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 2) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[3]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 3) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[0]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 4) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[4] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 5) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[5] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 6) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[6] + cell[idx].valueOfVertex[7]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 7) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[7] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 8) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[0] + cell[idx].valueOfVertex[4]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 9) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[1] + cell[idx].valueOfVertex[5]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 10) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[2] + cell[idx].valueOfVertex[6]) * 0.5f;
		if (d_triTable[(usage * 16) + (i * 3) + 2] == 11) cell[idx].triangles[i].density[2] = (cell[idx].valueOfVertex[3] + cell[idx].valueOfVertex[7]) * 0.5f;
	}
}

__global__ void add_triangle_to_array(Cell* cell, Triangle* triangleArr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int amount = cell[idx].triangleCnt;
	triangleArr[idx * 5 + 0] = cell[idx].triangles[0];
	triangleArr[idx * 5 + 1] = cell[idx].triangles[1];
	triangleArr[idx * 5 + 2] = cell[idx].triangles[2];
	triangleArr[idx * 5 + 3] = cell[idx].triangles[3];
	triangleArr[idx * 5 + 4] = cell[idx].triangles[4];
}

__global__ void interpolate_cell_density(Cell* cell) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
}

void MarchingCube::compute_cell_bit(float isoValue)
{
	compute_bit << <((axisX + 1) * (axisY + 1) * (axisZ + 1)) / 64, 64 >> > (d_data.cells, axisX, axisY, axisZ, isoValue);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else {
		printf("CUDA Success\n");
	}
	printf("inside compute_cell_bit\n");
}

void MarchingCube::make_triangle(float isoValue)
{
	make_cell_triangle <<< ((axisX) * (axisY) * (axisZ)) / 64, 64 >> > (d_data.cells, d_data.edgeTable, d_data.triangleTable, axisX, axisY, axisZ, isoValue);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else {
		printf("CUDA Success\n");
	}
	printf("inside make_cell_triangle\n");
}

void MarchingCube::make_triangle_arr()
{
/*
	int d_idx = 0;
	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				cudaMemcpy(&(h_data.cells[i][j][k]), &(d_data.cells[d_idx]), sizeof(Cell), cudaMemcpyDeviceToHost);
				d_idx++;
			}
		}
	}
*/

	printf("tssdf23423523523etset\n");
	cudaMemcpy(h_data.cells, d_data.cells, sizeof(Cell)* axisX* axisY* axisZ, cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else {
		printf("CUDA Success\n");
	}
	printf("tsetsetsetset\n");
	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				for (int l = 0; l < h_data.cells[(i*axisY*axisZ)+(j*axisZ)+k].triangleCnt; ++l)
				{
					h_data.triangles .push_back(h_data.cells[(i * axisY * axisZ) + (j * axisZ) + k].triangles[l]);
				}
			}
		}
	}
}

void MarchingCube::alloc_device_memory()
{
	int d_idx = 0;
	printf("inside alloc device memory\n");
	cudaMalloc((void**)&(d_data.cells), axisX * axisY * axisZ * sizeof(Cell));
	cudaMalloc((void**)&d_data.edgeTable, 256 * sizeof(int));
	cudaMalloc((void**)&d_data.triangleTable, 256 * 16 * sizeof(short int));
	cudaMalloc((void**)&d_data.triangles, axisX * axisY * axisZ * sizeof(Triangle));

	cudaMemcpy(d_data.edgeTable, h_data.edgeTable, 256 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data.triangleTable, h_data.triangleTable, 256 * 16 * sizeof(short int), cudaMemcpyHostToDevice);
/*
	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				cells[i][j][k].index = d_idx;
				cudaMemcpy(&d_cells[d_idx], &cells[i][j][k], sizeof(Cell), cudaMemcpyHostToDevice);
				d_idx++;
			}
		}
	}    
*/

	cudaMemcpy(d_data.cells, h_data.cells, sizeof(Cell) * axisX * axisY * axisZ, cudaMemcpyHostToDevice);
	printf("inside alloc device memory\n");
}

void MarchingCube::free_device_memory()
{
	cudaFree(d_data.cells);
	cudaFree(d_data.edgeTable);
	cudaFree(d_data.triangleTable);
	cudaFree(d_data.triangles);
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
	}

	fclose(file);

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
	float minDensity = particles[0].density;
	float maxDensity = particles[0].density;
	for (int i = 0; i < particleSize; ++i)
	{
		if (particles[i].density < minDensity) minDensity = particles[i].density;
		if (particles[i].density > maxDensity) maxDensity = particles[i].density;
	}

	printf("minDensity = %f\n", minDensity);
	printf("maxDensity = %f\n", maxDensity);

	fclose(file2);

	return true;
}

bool MarchingCube::make_polygon_with_particles(std::vector<vec3> vertices, float isoValue)
{
	particles = new Particle[vertices.size()];
	particleSize = vertices.size();

	for (int i = 0; i < vertices.size(); ++i)
	{
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

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else {
		printf("CUDA Success\n");
	}
	printf("inside make_cell_triangle\n");
	return true;
}

bool MarchingCube::generate_grid()
{
	find_grid_minmax();

	vec3 tmpVertex = maxVertex - minVertex;
	gridSize = std::min(tmpVertex.x, std::min(tmpVertex.y, tmpVertex.z)) / 40;
	
	axisX = (int(tmpVertex.x / gridSize) + 3);
	axisY = (int(tmpVertex.y / gridSize) + 3);
	axisZ = (int(tmpVertex.z / gridSize) + 3);

	printf("%d, %d, %d\n", axisX, axisY, axisZ);
	initialize_cell();
	printf("size of cell array = %d\n", sizeof(Cell) * axisX * axisY * axisZ);

	return true;
}

bool MarchingCube::put_density_into_cell()
{
	for (int i = 0; i < particleSize; ++i)
	{
		h_data.cells[(int((particles[i].position.x - minVertex.x) / gridSize)*axisY*axisZ)+(int((particles[i].position.y - minVertex.y) / gridSize)*axisZ) + int((particles[i].position.z - minVertex.z) / gridSize)].particleCnt++;
	}

	for (int i = 0; i < particleSize; ++i)
	{
		h_data.cells[(int((particles[i].position.x - minVertex.x) / gridSize) * axisY * axisZ) + (int((particles[i].position.y - minVertex.y) / gridSize) * axisZ) + int((particles[i].position.z - minVertex.z) / gridSize)].density += particles[i].density / h_data.cells[(int((particles[i].position.x - minVertex.x) / gridSize) * axisY * axisZ) + (int((particles[i].position.y - minVertex.y) / gridSize) * axisZ) + int((particles[i].position.z - minVertex.z) / gridSize)].particleCnt;
	}
	return true;
}

bool MarchingCube::initialize_cell()
{
	h_data.cells = new Cell[axisX * axisY * axisZ];

	for (int i = 0; i < axisX; ++i)
	{
		for (int j = 0; j < axisY; ++j)
		{
			for (int k = 0; k < axisZ; ++k)
			{
				h_data.cells[(i * axisY * axisZ) + (j * axisZ) + k].coordinate = vec3{
					minVertex.x + (gridSize / 2) + gridSize * i,
					minVertex.y + (gridSize / 2) + gridSize * j,
					minVertex.z + (gridSize / 2) + gridSize * k
				};

				h_data.cells[(i * axisY * axisZ) + (j * axisZ) + k].set_vertex_with_coordinate(gridSize);
			}
		}
	}

	//h_triangles = new Triangle[axisX * axisY * axisZ];

	return true;
}

bool MarchingCube::get_vertices_by_vtk(std::string filepath)
{
	std::string line;
	char* testLine;
	std::ifstream file(filepath);

	if (!file.is_open()) {
		std::cerr << "파일을 열 수 없습니다." << std::endl;
		return 1;
	}

	while (std::getline(file, line)) {
		if (line.find("POLYDATA") != std::string::npos) {
			std::cout << line << std::endl;
			break;
		}
	}
	std::string tmpStr;
	file >> tmpStr;
	int numFloats;
	file >> numFloats;
	file >> tmpStr;
	float* tmpFloat;

	std::cout << numFloats << std::endl;

	tmpFloat = new float[numFloats];

	for (int i = 0; i < numFloats; ++i)
	{
		tmpFloat[i] = 0.0;
		file >> tmpFloat[i];
		std::cout << tmpFloat[i] << std::endl;
	}
	/*
	std::cout << numFloats << std::endl;

	while (std::getline(file, line)) {
		if (line.find("POINTS") != std::string::npos) {
			std::cout << line << std::endl;
			break;
		}
	}
	*/
	//std::getline(file, line);
	//std::cout << line << std::endl;

	/*
	std::vector<unsigned char> floatData(numFloats * sizeof(unsigned char));

	file.read(reinterpret_cast<char*>(floatData.data()), numFloats * sizeof(unsigned char));

	//file.read()
	for (unsigned char value : floatData) {
		std::cout << static_cast<int>(value) << " ";
	}
	
	*/

	//std::vector<float> floatData(numFloats);
	//file.read(reinterpret_cast<char*>(floatData.data()), numFloats * sizeof(float));

	//for (float value : floatData) {
	//	std::cout << value << std::endl;
	//}

	file.close();
	
	/*
	if (file.is_open()) {
		while (std::getline(file, line)) {
			if (isInsidePoint) {
				file.read(testLine, 282900 * sizeof(float));
				printf("%s\n", testLine);
			}
			if (line.find("POINTS") != std::string::npos) {
				isInsidePoint = true;
			}
		}
	}
	*/
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
	printf("min : (%f, %f, %f)\n", minVertex.x, minVertex.y, minVertex.z);
	printf("max : (%f, %f, %f)\n", maxVertex.x, maxVertex.y, maxVertex.z);
	return true;
}

void MarchingCube::print_txt(std::string filepath)
{
	FILE* file = NULL;

	fopen_s(&file, filepath.c_str(), "wb");

	for (int i = 0; i < h_data.triangles.size(); ++i) {

		fwrite(&h_data.triangles[i].t1.x, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t1.y, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t1.z, sizeof(float), 1, file);

		fwrite(&h_data.triangles[i].t2.x, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t2.y, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t2.z, sizeof(float), 1, file);

		fwrite(&h_data.triangles[i].t3.x, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t3.y, sizeof(float), 1, file);
		fwrite(&h_data.triangles[i].t3.z, sizeof(float), 1, file);
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
	txt = "<Piece NumberOfPoints=\"" + std::to_string(h_data.triangles.size() * 3) + "\" NumberOfCells=\"" + std::to_string(h_data.triangles.size()) + "\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "<Points>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "<DataArray type=\"Float64\" NumberOfComponents=\"" + std::to_string(3) + "\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	txt = "";
	for (int i = 0; i < h_data.triangles.size(); ++i) {
		txt += (std::to_string(h_data.triangles[i].t1.x) + " " + std::to_string(h_data.triangles[i].t1.y) + " " + std::to_string(h_data.triangles[i].t1.z) + "\n");
		txt += (std::to_string(h_data.triangles[i].t2.x) + " " + std::to_string(h_data.triangles[i].t2.y) + " " + std::to_string(h_data.triangles[i].t2.z) + "\n");
		txt += (std::to_string(h_data.triangles[i].t3.x) + " " + std::to_string(h_data.triangles[i].t3.y) + " " + std::to_string(h_data.triangles[i].t3.z) + "\n");
	}

	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</Points>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<Cells>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "";

	for (int i = 0; i < h_data.triangles.size(); ++i)
	{
		txt += std::to_string(i * 3) + " ";
	}
	txt += "\n";

	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "";
	for (int i = 0; i < h_data.triangles.size()*3; ++i)
	{
		if (i % 3 == 2) txt += (std::to_string(i) + "\n");
		else txt += (std::to_string(i) + " ");
	}
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "";
	for (int i = 0; i < h_data.triangles.size(); ++i)
	{
		txt += "5 ";
	}
	txt += "\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n</Cells>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "<PointData>\n<DataArray type = \"Float32\" Name=\"Density\" format=\"ascii\">\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "";

	for (int i = 0; i < h_data.triangles.size(); ++i)
	{
		txt += std::to_string(h_data.triangles[i].density[0]) + " ";
		txt += std::to_string(h_data.triangles[i].density[1]) + " ";
		txt += std::to_string(h_data.triangles[i].density[2]) + " ";
	}
	txt += "\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</DataArray>\n</PointData>\n";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);

	txt = "</Piece>\n</UnstructuredGrid>\n</VTKFile>";
	fwrite(txt.c_str(), sizeof(char), txt.size(), file);
	fclose(file);
}

void MarchingCube::set_density() {

}
void MarchingCube::set_vertices()
{

}

void MarchingCube::write_binary(std::string txt)
{

}