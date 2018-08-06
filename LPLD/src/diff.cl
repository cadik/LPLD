#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void diff(
    const __global real_t * in_test_img,
    const __global real_t * in_ref_img,
    __global real_t * output,
    __constant real_t * filter,
    __local real_t * cached_test,
    __local real_t * cached_ref
){



}