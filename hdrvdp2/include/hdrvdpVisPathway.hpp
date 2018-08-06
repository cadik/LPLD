#ifndef DIP_HDRVDP_VIS_PATHWAY_HPP
#define DIP_HDRVDP_VIS_PATHWAY_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <stdexcept>
#include <iostream>
#include <CL/cl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "fftw3.h"

#include "GeneralPurposeLib.hpp"
#include "csvIterator.hpp"
#include "metric_params.hpp"
#include "hdrvdpHelper.hpp"
#include "clClass.hpp"

extern "C" {
#include "spyramid.h"
}

namespace dip {
    class HDRVDP_VisPathway {
    public:
        HDRVDP_VisPathway(const cv::Mat &image, const MetricParams &metric_par, clClass *clObj);
        ~HDRVDP_VisPathway() {delete[] band_freq;
                              delete[] bands_sz;
        }
        
        cv::Mat L_adapt;
        double *band_freq;
        int *bands_sz;
        int bandSize;
        int bandsSum;
        PYRAMID P_pyramid;
        PYRAMID D_bands;

        static cv::Mat hdrvdp_ncsf(const cv::Mat &rho, double lum, const MetricParams &metric_par);
        static cv::Mat hdrvdp_ncsf(double rho, const cv::Mat &lum, const MetricParams &metric_par);

    private:
        unsigned int m_imgWidth;
        unsigned int m_imgHeight;
        unsigned int m_imgChannels;
        
        std::vector<double> lambda;
        
        cv::Mat m_rho2;
        cv::Mat m_mtf_filter;
        cv::Mat LMSR_S;
        cv::Mat L_O;
        cv::Mat M_img_lmsr;
        cv::Mat R_LMSR;
        cv::Mat pn_jnd[2];
        cv::Mat pn_Y;

        clClass *m_clObj;

        cv::Mat CreateCycdegImage(int imgWidth, int imgHeight, double pix_per_deg);
        void CreateMtf(const MetricParams &metric_par);
        void load_LMSR_S() throw (std::runtime_error);
        void photorecNonLinear(const MetricParams &metric_par);
        cv::Mat hdrvdp_rod_sens(const cv::Mat &la, const MetricParams &metric_par);
        cv::Mat build_jndspace_from_S(const cv::Mat &l, const cv::Mat &S);
        void opticalTransferFun(const cv::Mat &img, const MetricParams &metric_par);
        cv::Mat fft_convolution(const cv::Mat &image, const cv::Mat &filter, double padValue);
        void initLambda();
        void photoSpectralSensitivity(const MetricParams &metric_par);
        double trapz(std::vector<double> &x, std::vector<double> &y);
        void fastElementWiseMul(const cv::Mat &in1, const cv::Mat &in2, cv::Mat &dst);
		std::vector<double> linear_interpolation_GPU(std::map<double, double> &X_V, int cycles, double * ptrToData);
		std::vector<double> getXvec_GPU(double* ptrToData, int cycles, int size);

        int defaultLevelsNum(int imgCols, int imgRows, int filterSize);
    };
}

#endif
