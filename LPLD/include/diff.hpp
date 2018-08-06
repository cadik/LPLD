#ifndef DIP_DIFF_HPP
#define DIP_DIFF_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "CLSharedLib.hpp"
#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"

#include <CL/cl.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdafx.h"

namespace dip {
    class DIFF : public methodStrategy {
    public:
        DIFF(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error);
        void compute() throw (std::runtime_error);

        const cv::Mat getResult() {return m_result;}

    private:
		void initCL()throw (std::runtime_error, cl::Error);
		template<typename T> void private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error);
        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        cv::Mat m_result;

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
