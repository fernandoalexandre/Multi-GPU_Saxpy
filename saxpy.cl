/**
  * Created by Ricardo Marques
 **/

__kernel void saxpy(__global float *X, __global float *Y, const float a, __global float *out)
{
	int pos = get_global_id(0);
	float x = X[pos];
	float y = Y[pos];
	out[pos] = a * x + y;
}
