#ifndef DIP_HARRIS_HPP
#define DIP_HARRIS_HPP

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
    #include <vl/imopv.h>
}

namespace dip {
    class HARRIS : public methodStrategy {
    public:
        HARRIS(cv::Mat &testImg, cv::Mat &referenceImg, int level) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result.at(m_level); }

    private:
        cv::Mat computeHarris(const cv::Mat &inputMat, const cv::Mat &gradX, const cv::Mat &gradY, double sigma);

        cv::Mat gradientX(const cv::Mat & mat, double spacing);
        cv::Mat gradientY(const cv::Mat & mat, double spacing);
        cv::Mat smoothImage(const cv::Mat &image, const double sigma);

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        int m_level;
        std::vector<cv::Mat> m_result;
    };
}

#endif
