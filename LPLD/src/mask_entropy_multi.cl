#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void mask_entropy_multi(
	__global real_t * image,
	__global real_t * out

)
{
	int r = get_global_id(1);
	int c = get_global_id(0);

	real_t roi_heap[49];
	int window_size = WINDOWSIZE;

	for(int i = 0; i < window_size; i++){
		for(int j = 0; j < window_size; j++){
			roi_heap[i + j*window_size] = image[c+i + (r+j)*PADDED_COLS];

		}
	}

	int histogram[256];
	for(int i = 0; i < 256; i++){
		histogram[i] = 0;
	}
	for(int i = 0; i < window_size; i++){
		for(int j = 0; j < window_size; j++){
			int temp =roi_heap[i + j*window_size];
			histogram[temp] = histogram[temp] + 1;
		}
	}
	real_t sum = 0;
	for(int i=0; i < 256; i++){
		if(histogram[i] != 0){
			real_t histValue = histogram[i];
			sum += (histValue/ (window_size*window_size)) * (log(histValue / (window_size*window_size)) / log(2.0));
		}
	}
	
	out[c + r*COLS] = -1.0 * sum;
}
