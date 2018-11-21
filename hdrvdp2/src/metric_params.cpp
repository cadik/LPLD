#include "metric_params.hpp"

namespace dip {
    double MetricParams::csf_params[6][5] = {{0.0164124, 1.00195,  3.54717, 1.08514, 3.535}, \
                                            {0.379874,  0.795286, 3.24414, 1.45004, 3.60714},\
                                            {0.919402,  0.423683, 4.22778, 1.56528, 3.62531},\
                                            {1.28364,   0.348077, 4.24637, 1.96683, 3.62059},\
                                            {1.48094,   0.246298, 3.85941, 2.24956, 3.7662},\
                                            {1.46903,   0.209051, 3.06441, 2.34258, 3.58503}};

    double MetricParams::csf_lums[6] = {0.002, 0.02, 0.2, 2, 20, 150};
    double MetricParams::csf_sa[4] = {30.182, 4.3806, 1.5154, 0.29412};
    double MetricParams::csf_sr_par[6] = {1.1732, 1.1478, 1.2167, 0.5547, 2.9899, 1.1};

    float MetricParams::csf_sa_f[4] = { 30.182f, 4.3806f, 1.5154f, 0.29412f };
    float MetricParams::csf_sr_par_f[6] = { 1.1732f, 1.1478f, 1.2167f, 0.5547f, 2.9899f, 1.1f };

    double MetricParams::par[2] = {0.061466549455263, 0.99727370023777070};
    double MetricParams::mtf_params_a[4] = {par[1]*0.426, par[1]*0.574, (1-par[1])*par[0], (1-par[1])*(1-par[0])};
    double MetricParams::mtf_params_b[4] = {0.028, 0.37, 37, 360};

    float MetricParams::par_f[2] = { 0.061466549455263f, 0.99727370023777070f };
    float MetricParams::mtf_params_a_f[4] = { par_f[1] * 0.426f, par_f[1] * 0.574f, (1.0f - par_f[1])*par_f[0], (1.0f - par_f[1])*(1.0f - par_f[0]) };
    float MetricParams::mtf_params_b_f[4] = { 0.028f, 0.37f, 37.0f, 360.0f };

    double MetricParams::quality_band_freq[7] = {15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344};
    double MetricParams::quality_band_w[7] = {0.2963, 0.2111, 0.1737, 0.0581, -0.0280, 0.0586, 0.2302};

    MetricParams::MetricParams() {
        // Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
        this->daly_peak_contrast_sens = 0.006894596;
        this->sensitivity_correction = daly_peak_contrast_sens / pow(10.0, -2.4); 
        this->view_dist = 0.5;

        //this->spectral_emission = [];

        this->orient_count = 4; // the number of orientations to consider

        // Various optional features
        this->do_masking = true;
        this->do_mtf = true;
        this->do_spatial_pooling = true;
        this->noise_model = true;
        this->do_quality_raw_data = false; // for development purposes only

        this->steerpyr_filter = "sp3Filters";

        this->mask_p = 0.544068;
        this->mask_self = 0.189065;
        this->mask_xo = 0.449199;
        this->mask_xn = 1.52512;
        this->mask_q = 0.49576;

        this->psych_func_slope = log10(3.5);
        this->beta = psych_func_slope - mask_p;

        // Spatial summation
        this->si_slope = -0.850147;
        this->si_sigma = -0.000502005;
        this->si_ampl = 0;

        // Cone and rod cvi functions
        this->cvi_sens_drop = 0.0704457;
        this->cvi_trans_slope = 0.0626528;
        this->cvi_low_slope = -0.00222585;

        this->rod_sensitivity = 0;
        this->rod_sensitivity_f = 0.0f;
        //this->rod_sensitivity = -0.383324;
        this->cvi_sens_drop_rod = -0.58342;

        // Achromatic CSF
        this->csf_m1_f_max = 0.425509;
        this->csf_m1_s_high = -0.227224;
        this->csf_m1_s_low = -0.227224;
        this->csf_m1_exp_low = log10( 2 );

        // Daly CSF model
        this->csf_stim_area = 0; //pow(2.5,2);
        this->csf_epsilon = -0.546385;
        this->csf_peak_shift = 0.235954;
        this->csf_lf_slope = -0.844601;
        this->csf_peak_shift_lum = 1.16336;
        this->csf_peak_shift_slope = -0.912733;

        // Fix for the non-linearity after cortex transform
        this->ibf_fix = log10(0.321678);

        // Rod CSF
        this->csf_rod_f_max = 0.15;
        this->csf_rod_s_low = -0.266858;    
        this->csf_rod_exp_low = log10(2);    
        this->csf_rod_s_high = -0.266858;   

        this->quality_logistic_q1 = 4.446;
        this->quality_logistic_q2 = 0.8994;

        this->surround_l = -1; //use mean image luminance
    }
}