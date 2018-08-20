#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void display_sRGB( __global double* imageIn1, 
                            __global double* imageIn2) {

    __private double a = 0.055;
    __private double thr = 0.04045;

    int i = get_global_id(0);

    __private double value1 = imageIn1[i];
    __private double value2 = imageIn2[i];

    if (value1 > thr)
        imageIn1[i] = 99.0 * pow((value1 + a) / (1.0 + a), 2.4) + 1.0;
    else
        imageIn1[i] = 99.0 * (value1 / 12.92) + 1.0;

    if (value2 > thr)
        imageIn2[i] = 99.0 * pow((value2 + a) / (1.0 + a), 2.4) + 1.0;
    else
        imageIn2[i] = 99.0 * (value2 / 12.92) + 1.0;

    return;
}


__kernel void createSpace(double d1, double d2, \
                          double n, __global double *result, int type) {
    
    int i = get_global_id(0);
    double n1 = floor(n)-1;
    
    double value = d1 + i * (d2 - d1)/n1;

    if (type == 1)
        result[i] = value;
    else
        result[i] = pown(value, 10);

    if (i == n1) {
        result[0] = d1;
        result[i] = d2;
    }

    return;
}



__kernel void hdrvdp_mtfCL(__global double *rho, __global double*result, __global double* params_A, __global double* params_B) {
    int pos = get_global_id(0);

    result[pos] = result[pos] + params_A[0] * exp(-params_B[0] * rho[pos]);
    result[pos] = result[pos] + params_A[1] * exp(-params_B[1] * rho[pos]);
    result[pos] = result[pos] + params_A[2] * exp(-params_B[2] * rho[pos]);
    result[pos] = result[pos] + params_A[3] * exp(-params_B[3] * rho[pos]);

    return;
}


__kernel void jointRodConeSens_rodSens(__constant double *la, \
                                       __global   double *s_A, \
                                       __global   double *s_R, \
                                       __global   double *V, \
                                       __global   double *Xq, \
                                       __constant double *csf_sa, \
                                       __constant double *csf_sr_par, \
                                                  double rod_sensitivity) {

    int pos = get_global_id(0);
    int lastIndex = get_global_size(0) - 1;

    double cvi_sens_drop = csf_sa[1];
    double cvi_trans_slope = csf_sa[2];
    double cvi_low_slope = csf_sa[3];

    double peak_l = csf_sr_par[0];
    double low_s = csf_sr_par[1];
    double low_exp = csf_sr_par[2];
    double high_s = csf_sr_par[3];
    double high_exp = csf_sr_par[4];
    double rod_sens = csf_sr_par[5];

    s_A[pos] =  csf_sa[0] * pow(pow((cvi_sens_drop/la[pos]), cvi_trans_slope) + 1.0, -cvi_low_slope);

    if (la[pos] > peak_l)
        s_R[pos] = exp( -pow(fabs(log10(la[pos]/peak_l)), high_exp) / high_s ) * rod_sens;
    else
        s_R[pos] = exp( -pow(fabs(log10(la[pos]/peak_l)), low_exp) / low_s ) * rod_sens;



    s_R[pos] = (s_R[pos] * pow(10.0, rod_sens)) * pow(10.0, rod_sensitivity);

    //0.5 * interp1( c_l, max(s_A-s_R, 1e-3), min( c_l*2, c_l(end) ) );
    V[pos] = max(s_A[pos] - s_R[pos] , 1e-3);
    Xq[pos] = min(la[pos] * 2.0 , la[lastIndex]);

    return;
}


__kernel void hdrvdp( __global double4* imageIn_1) {


    return;
}

