// Based on source code for convolution available in book:
// Heterogeneous Computing with OpenCL
// ISBN: 978-0-12-387766-6

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif


__kernel void ssim_lum(
    const __global real_t * in_test_img,
    const __global real_t * in_ref_img,
    __global real_t * output,
    __constant real_t * filter,
    __local real_t * cached_test,
    __local real_t * cached_ref
)
{
    const int rowOffset = get_global_id(1) * IMAGE_W;
    const int my = get_global_id(0) + rowOffset;

    const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);
    const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen;
    const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;

    // copy my pixel
    cached_test[ myLocal ] = in_test_img[ my ];
    cached_ref[ myLocal ] = in_ref_img[ my ];

    if (
        get_global_id(0) < HALF_FILTER_SIZE             ||
        get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1       ||
        get_global_id(1) < HALF_FILTER_SIZE         ||
        get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
    )
    {
        // no computation for me, sync and exit
        barrier(CLK_LOCAL_MEM_FENCE);
        return;
    }
    else
    {
        // copy additional elements
        int localColOffset = -1;
        int globalColOffset = -1;

        if ( get_local_id(0) < HALF_FILTER_SIZE )
        {
            localColOffset = get_local_id(0);
            globalColOffset = -HALF_FILTER_SIZE;

            cached_test[ localRowOffset + get_local_id(0) ] = in_test_img[ my - HALF_FILTER_SIZE ];
            cached_ref[ localRowOffset + get_local_id(0) ] = in_ref_img[ my - HALF_FILTER_SIZE ];
        }
        else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
        {
            localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
            globalColOffset = HALF_FILTER_SIZE;

            cached_test[ myLocal + HALF_FILTER_SIZE ] = in_test_img[ my + HALF_FILTER_SIZE ];
            cached_ref[ myLocal + HALF_FILTER_SIZE ] = in_ref_img[ my + HALF_FILTER_SIZE ];
        }


        if ( get_local_id(1) < HALF_FILTER_SIZE )
        {
            cached_test[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = in_test_img[ my - HALF_FILTER_SIZE_IMAGE_W ];
            cached_ref[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = in_ref_img[ my - HALF_FILTER_SIZE_IMAGE_W ];
            if (localColOffset > 0)
            {
                cached_test[ get_local_id(1) * localRowLen + localColOffset ] = in_test_img[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
                cached_ref[ get_local_id(1) * localRowLen + localColOffset ] = in_ref_img[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
            }
        }
        else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
        {
            int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
            cached_test[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = in_test_img[ my + HALF_FILTER_SIZE_IMAGE_W ];
            cached_ref[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = in_ref_img[ my + HALF_FILTER_SIZE_IMAGE_W ];
            if (localColOffset > 0)
            {
                cached_test[ offset + localColOffset ] = in_test_img[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
                cached_ref[ offset + localColOffset ] = in_ref_img[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
            }
        }

        // sync
        barrier(CLK_LOCAL_MEM_FENCE);

        // perform convolution
        int fIndex = 0;
        real_t mu1 = 0.0;
        real_t mu2 = 0.0;

        for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
        {
            int curRow = r * localRowLen;
            for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)
            {
                mu1 += cached_test[ myLocal + curRow + c ] * filter[ fIndex ];
                mu2 += cached_ref[ myLocal + curRow + c ] * filter[ fIndex ];
            }
        }


        real_t mu1_sq =  mu1 * mu1;
        real_t mu2_sq =  mu2 * mu2;
        real_t mu1_mu2 = mu1 * mu2;

        real_t lum = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1);

        output[my] =  lum;
    }
}

