#ifndef DIP_HOG9_HPP
#define DIP_HOG9_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include "GeneralPurposeLib.hpp"
#include "algStrategy.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ocl/ocl.hpp>

extern "C" {
#include "vl/hog.h"
}

#include "stdafx.h"

namespace dip {
    class HOG9 : public methodStrategy {
    public:
        HOG9(cv::Mat &testImg, cv::Mat &referenceImg) throw (std::runtime_error);
        void compute() throw (std::runtime_error);
        const cv::Mat getResult() {return m_result; }

    private:
        cv::Mat get_hogdescriptor(const cv::Mat &inputMat);

        cv::Mat m_ref_img;
        cv::Mat m_test_img;

        cv::Mat m_result;
    };
}

#endif
