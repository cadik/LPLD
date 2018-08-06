#ifndef DIP_SPYR_DIST_HPP
#define DIP_SPYR_DIST_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stdafx.h"

extern "C" {
#include "spyramid.h"
}

namespace dip {
    class SPYR_DIST : public methodStrategy {
    public:
        SPYR_DIST(cv::Mat &testImg, cv::Mat &referenceImg) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result; }

    private:
        std::vector<cv::Mat> vis_dis(const cv::Mat &inputMat);
        cv::Mat get_band(PYRAMID pyr, int band, int orienation);
        unsigned int roundUp(unsigned int value, unsigned int multiple) const;
        cv::Mat computeSTD(cv::Mat &inputMat);

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        cv::Mat m_result;
    };
}

#endif
