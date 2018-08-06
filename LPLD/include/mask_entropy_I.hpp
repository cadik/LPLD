#ifndef DIP_MASK_ENTROPY_I_HPP
#define DIP_MASK_ENTROPY_I_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"
#include "CLSharedLib.hpp"
#include <CL/cl.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdafx.h"

namespace dip {
    class MASK_ENT_I : public methodStrategy {
    public:
        MASK_ENT_I(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result; }

    private:
		void initCL() throw (std::runtime_error, cl::Error);
		void computeEntropy_GPU(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize);
        void computeEntropy(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize);

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        cv::Mat m_result;

		cl::Platform m_platform;
		cl::Device m_device;
		cl::Context m_context;
		cl::CommandQueue m_queue;
		cl::Program m_program;

		int m_platformNum;
		cl_device_type m_deviceType;
		int m_deviceNum;
		std::string m_clSrcFileName;

		std::vector<cl::Device> m_devicesVector;

		bool m_dev_doubleSupport;

		// Size of a work group
		cl::NDRange m_localSize;
		// Size of the NDRange
		cl::NDRange m_globalSize;
    };
}

#endif
