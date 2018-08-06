#ifndef DIP_HDRVDP_BAND_I_HPP
#define DIP_HDRVDP_BAND_I_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "GeneralPurposeLib.hpp"
#include "hdrvdp.hpp"
#include "algStrategy.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdafx.h"

namespace dip {
    class HDRVDP_BAND : public methodStrategy {
    public:
		HDRVDP_BAND(cv::Mat &testImg, cv::Mat &referenceImg, int level) throw (std::runtime_error);
		HDRVDP_BAND(cv::Mat &testImg, cv::Mat &referenceImg, int level, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result.at(m_level); }

    private:
        void loadSources(const std::string &reference_img, const std::string &test_img);
        void computeResult() throw (std::runtime_error);

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        int m_level;
        std::vector<cv::Mat> m_result;

		int m_platformNum;
		cl_device_type m_deviceType;
		int m_deviceNum;
    };
}

#endif
