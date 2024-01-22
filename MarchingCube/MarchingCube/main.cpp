#include "testor.h"

int main()
{
	Testor myTestor;
	vec3 a = { 1.0,-5.0,3.0 };
	vec3 b = { -5.0,3.0,9.0 };
	vec3 c = a - b;
	printf("x = %f\n", c.x);
	printf("y = %f\n", c.y);
	printf("z = %f\n", c.z);
	return 0;
}