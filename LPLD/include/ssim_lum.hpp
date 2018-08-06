#ifndef DIP_SSIM_LUM_HPP
#define DIP_SSIM_LUM_HPP

#ifdef _MSC_VER
    #pragma warning(disable : 4290)
#endif

#define __CL_ENABLE_EXCEPTIONS

#include "CLSharedLib.hpp"
#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"

#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fftw3.h>

#include <CL/cl.hpp>

namespace dip {
    class SSIM_LUM : public methodStrategy {
    public:
        // Constructor
        SSIM_LUM(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum = 0, \
                    cl_device_type deviceType = CL_DEVICE_TYPE_GPU, int deviceNum = 0) throw (std::runtime_error);

        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result; }

    private:
        void initCL() throw(std::runtime_error, cl::Error);
        template<typename T> void private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error);
        double fftw_convolution();

        unsigned int roundUp(unsigned int value, unsigned int multiple) const;
        inline int nextPowerOf2(int32_t number) const;
        cv::Mat normalizeMat(const cv::Mat& inputMat);

        cv::Mat m_test_Img;
        cv::Mat m_reference_Img;
        cv::Mat m_result;

        size_t m_image_width;
        size_t m_image_height;

        int m_platformNum;
        cl_device_type m_deviceType;
        int m_deviceNum;
        std::string m_clSrcFileName;

        std::vector<cl::Device> m_devicesVector;

        // Size of a work group
        cl::NDRange m_localSize;
        // Size of the NDRange
        cl::NDRange m_globalSize;

        bool m_dev_doubleSupport;

        double m_sigma;
        double m_k_1;
        double m_k_2;
        size_t m_filterSize;

        cl::Platform m_platform;
        cl::Device m_device;
        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
    };
}

#endif
