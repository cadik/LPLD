#ifndef DIP_GRAD_DIST_HPP
#define DIP_GRAD_DIST_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "CLSharedLib.hpp"
#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"
#include <CL/cl.hpp>
#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dip {
    class Grad_Dist : public methodStrategy {
    public:
        Grad_Dist(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum,	cl_device_type deviceType, int deviceNum) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result; }

    private:
		void initCL() throw (std::runtime_error, cl::Error);
		void vis_dist(cv::Mat &image, std::vector<cv::Mat> &dst, unsigned int rowBlock, unsigned int colBlock);
		void vis_dist_GPU(cv::Mat &image, std::vector<cv::Mat> &dst, unsigned int rowBlock, unsigned int colBlock);
        cv::Mat im2col(const cv::Mat &src, unsigned int rowBlock, unsigned int colBlock);

        cv::Mat gradientX(const cv::Mat & mat, double spacing);
        cv::Mat gradientY(const cv::Mat & mat, double spacing);
        
        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        int m_level;
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
