#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void grad_dist(
	__global real_t * grad1,
	__global real_t * grad2,
	__global real_t * out1,
	__global real_t * out2

)
{
	int r = get_global_id(1);
	int c = get_global_id(0);
    int lr = get_local_id(1);
	int lc = get_local_id(0);
	int wg = get_group_id(0) + get_group_id(1)*(COLS/8);
	
	__local real_t patch[64];
	__local real_t medianpatch[64];
	__local real_t patch2[64];
	__local real_t medianpatch2[64];

	if(lr == 0 && lc == 0){
		for(int i = 0; i < COLBLOCK; i++){
			for(int j = 0; j < ROWBLOCK; j++){
				if(c+i < COLS && r+j<ROWS){
					patch[i + COLBLOCK*j] = grad1[c+i + COLS*(r+j)];
					medianpatch[i + COLBLOCK*j] = grad1[c+i + COLS*(r+j)];
					patch2[i + COLBLOCK*j] = grad2[c+i + COLS*(r+j)];
					medianpatch2[i + COLBLOCK*j] = grad2[c+i + COLS*(r+j)];
				}else{
					patch[i + COLBLOCK*j] =0;
					medianpatch[i + COLBLOCK*j] =0;
					patch2[i + COLBLOCK*j] = 0;
					medianpatch2[i + COLBLOCK*j] = 0;
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int rank = lc+ lr*8;
	real_t tmp = 0;
	for(int i=0; i<64;i++){
		if(i % 2 == 0){
			if(lc % 2 == 0){
				if( medianpatch[rank]>medianpatch[rank+1]){
					tmp = medianpatch[rank+1];
					medianpatch[rank+1] =medianpatch[rank];
					medianpatch[rank] = tmp;
				}
			}else{
				if( medianpatch2[rank]<medianpatch2[rank-1]){
					tmp = medianpatch2[rank-1];
					medianpatch2[rank-1] =medianpatch2[rank];
					medianpatch2[rank] = tmp;
				}
			}
		}else{
			if(lc % 2 == 0){
				if(rank != 0){
					if( medianpatch[rank]<medianpatch[rank-1]){
						tmp = medianpatch[rank-1];
						medianpatch[rank-1] =medianpatch[rank];
						medianpatch[rank] = tmp;
					}
				}else{
					if( medianpatch[63]<medianpatch[0]){
						tmp = medianpatch[0];
						medianpatch[0] =medianpatch[63];
						medianpatch[63] = tmp;
					}
				}
			}else{
				if(rank != 63){
					if( medianpatch2[rank]>medianpatch2[rank+1]){
						tmp = medianpatch2[rank+1];
						medianpatch2[rank+1] =medianpatch2[rank];
						medianpatch2[rank] = tmp;
					}
				}else{
					if( medianpatch2[63]<medianpatch2[0]){
						tmp = medianpatch2[0];
						medianpatch2[0] =medianpatch2[63];
						medianpatch2[63] = tmp;
					}
				}
			}
		}
	barrier(CLK_LOCAL_MEM_FENCE);
	}
	__local real_t median;
	__local real_t median2;
	if(lr == 0 && lc == 0){
		median = (medianpatch[31] +medianpatch[32])/2;
		median2 = (medianpatch2[31] +medianpatch2[32])/2;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	patch[rank] = patch[rank]-median;
	patch2[rank] = patch2[rank]-median;
	if(patch[rank] < 0){
		patch[rank] = patch[rank] *(-1);
	}
	if(patch2[rank] < 0){
		patch2[rank] = patch2[rank] *(-1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	if(lr == 0 && lc == 0){
		real_t mean =0;
		for(int j = 0; j <  ROWBLOCK *COLBLOCK; j++){
			mean = mean + patch[j];
		}
		mean = mean /64;
		out1[wg]= mean;
	}else if(lr == 1 && lc == 1){
		real_t mean =0;
		for(int j = 0; j <  ROWBLOCK *COLBLOCK; j++){
			mean = mean + patch2[j];
		}
		mean = mean /64;
		out2[wg]= mean;
	}		
}
