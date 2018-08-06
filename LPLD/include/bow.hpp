#ifndef DIP_BOW_HPP
#define DIP_BOW_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#define NOMINMAX

#endif
#include "CLSharedLib.hpp"
#include "bow_params.hpp"
#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"

#include <CL/cl.hpp>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdafx.h"

namespace dip {
    class BOW : public methodStrategy {
    public:
        BOW(cv::Mat &testImg, cv::Mat &referenceImg, const BOW_PARAMS &params, int level, int platformNum, \
			cl_device_type deviceType, int deviceNum) throw (std::runtime_error)
            : m_params(params) {
            m_level = level;

			m_platformNum = platformNum;
			m_deviceNum = deviceNum;
			m_deviceType = deviceType;

            m_test_img = testImg;
            m_ref_img = referenceImg;

			m_image_width = m_test_img.cols;
			m_image_height = m_test_img.rows;

            if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
                throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
            if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
                throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

        }

		void compute() throw (std::runtime_error);
		void standard_compute() throw (std::runtime_error);
        const cv::Mat getResult() {return H_vec.at(m_level); }

    private:
		void BOW::initCL() throw (std::runtime_error, cl::Error);
		template<typename T> void private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error);
        BOW();
        std::vector<double> computeDCT_Decsriptor(int patchCord_X, int patchCord_Y, \
                                                  double randomRotation = 0.0);
        double getVariance(cv::Mat &inputMat, double meanVal);

        const BOW_PARAMS &m_params;
        cv::Mat m_lumRefImg;
        cv::Mat m_lumTestImg;
        cv::Mat m_diff;

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        int m_level;
        std::vector<cv::Mat> H_vec;

		size_t m_image_width;
		size_t m_image_height;
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
