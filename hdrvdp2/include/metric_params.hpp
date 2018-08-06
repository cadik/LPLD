#ifndef DIP_METRIC_PARAMS_HPP
#define DIP_METRIC_PARAMS_HPP

#include <string>
#include <cmath>
#include <opencv2/core/core.hpp>

namespace dip {
    class MetricParams {
    public:
        MetricParams();

        // Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
        double daly_peak_contrast_sens;
        double sensitivity_correction; 
        double view_dist;
        double pix_per_degree;

        cv::Mat spectral_emission;

        int orient_count; // the number of orientations to consider

        // Various optional features
        bool do_masking;
        bool do_mtf;
        bool do_spatial_pooling;
        bool noise_model;
        bool do_quality_raw_data; // for development purposes only


        std::string steerpyr_filter;

        double mask_p;
        double mask_self;
        double mask_xo;
        double mask_xn;
        double mask_q;

        double psych_func_slope;
        double beta;

        // Spatial summation
        double si_slope;
        double si_sigma;
        double si_ampl;

        // Cone and rod cvi functions
        double cvi_sens_drop;
        double cvi_trans_slope;
        double cvi_low_slope;

        double rod_sensitivity;
        float rod_sensitivity_f;
        //metric_par.rod_sensitivity;
        double cvi_sens_drop_rod;

        // Achromatic CSF
        double csf_m1_f_max;
        double csf_m1_s_high;
        double csf_m1_s_low;
        double csf_m1_exp_low;

        // Daly CSF model
        double csf_stim_area;
        double csf_epsilon;
        double csf_peak_shift;
        double csf_lf_slope;
        double csf_peak_shift_lum;
        double csf_peak_shift_slope;

        // Fix for the non-linearity after cortex transform
        double ibf_fix;

        // Rod CSF
        double csf_rod_f_max;
        double csf_rod_s_low;    
        double csf_rod_exp_low;    
        double csf_rod_s_high; 

        static double csf_params[6][5];
        static double csf_lums[6];
        static double csf_sa[4];
        static double csf_sr_par[6];

        static float csf_sa_f[4];
        static float csf_sr_par_f[6];

        static double par[2]; // old parametrization of MTF
        static double mtf_params_a[4];
        static double mtf_params_b[4];

        static float par_f[2];
        static float mtf_params_a_f[4];
        static float mtf_params_b_f[4];

        static double quality_band_freq[7];
        static double quality_band_w[7];

        double quality_logistic_q1;
        double quality_logistic_q2;

        double surround_l; //use mean image luminance

        void setSensitivitiCorrection(double value) {
            this->sensitivity_correction = this->daly_peak_contrast_sens / pow(10, value);
        }

    };
}


#endif