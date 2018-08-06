#include "harris.hpp"
#include <ctime>

namespace dip {
    HARRIS::HARRIS(cv::Mat &testImg, cv::Mat &referenceImg, int level) throw (std::runtime_error) {
        m_level = level;

        m_test_img = testImg;
        m_ref_img = referenceImg;

        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

    }


    void HARRIS::compute() throw (std::runtime_error) {
        m_test_img = GPLib::get_luminance(m_test_img);
        m_ref_img = GPLib::get_luminance(m_ref_img);
        double sigma = 0.5;

        cv::Mat Ix_test = gradientX(m_test_img, 1.0);
        cv::Mat Iy_test = gradientY(m_test_img, 1.0);

        cv::Mat Ix_ref = gradientX(m_ref_img, 1.0);
        cv::Mat Iy_ref = gradientY(m_ref_img, 1.0);

        for (int i = 0; i < 6; i++) {
            m_result.push_back(computeHarris(m_test_img, Ix_test, Iy_test, sigma));
            m_result.push_back(computeHarris(m_ref_img, Ix_ref, Iy_ref, sigma));
            
            sigma *= 2.0;
        }
    }

    cv::Mat HARRIS::computeHarris(const cv::Mat &inputMat, const cv::Mat &gradX, const cv::Mat &gradY, double sigma) {
        cv::Mat result(inputMat.rows, inputMat.cols, CV_64FC1);
        cv::Mat H11, H12, H22;

        H11 = smoothImage(gradX.mul(gradX), sigma);
        H12 = smoothImage(gradX.mul(gradY), sigma);
        H22 = smoothImage(gradY.mul(gradY), sigma);

        double *resultPtr = (double *) result.data;
        double *h11Ptr = (double *) H11.data;
        double *h12Ptr = (double *) H12.data;
        double *h22Ptr = (double *) H22.data;

        for (int i = 0; i < inputMat.cols * inputMat.rows; i++) {
            *resultPtr = 2 * ((*h11Ptr) * (*h22Ptr) - pow(*h12Ptr, 2.0)) / (*h11Ptr + *h22Ptr + DBL_EPSILON);
            
            resultPtr++;
            h11Ptr++; h12Ptr++; h22Ptr++;
        }

        return result;
    }


    cv::Mat HARRIS::gradientX(const cv::Mat & mat, double spacing) {
        cv::Mat grad = cv::Mat::zeros(mat.rows,mat.cols,CV_64FC1);

        /*  last row */
        int maxCols = mat.cols;
        int maxRows = mat.rows;

        /* get gradients in each border */
        /* first row */
        cv::Mat col = (-mat.col(0) + mat.col(1))/(double)spacing;
        col.copyTo(grad(cv::Rect(0,0,1,maxRows)));

        col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(double)spacing;
        col.copyTo(grad(cv::Rect(maxCols-1,0,1,maxRows)));

        /* centered elements */
        cv::Mat centeredMat = mat(cv::Rect(0,0,maxCols-2,maxRows));
        cv::Mat offsetMat = mat(cv::Rect(2,0,maxCols-2,maxRows));
        cv::Mat resultCenteredMat = (-centeredMat + offsetMat)/(((double)spacing)*2.0);

        resultCenteredMat.copyTo(grad(cv::Rect(1,0,maxCols-2, maxRows)));
        return grad;
    }


    cv::Mat HARRIS::gradientY(const cv::Mat & mat, double spacing) {
        cv::Mat grad = cv::Mat::zeros(mat.rows,mat.cols,CV_64FC1);

        /*  last row */
        const int maxCols = mat.cols;
        const int maxRows = mat.rows;

        /* get gradients in each border */
        /* first row */
        cv::Mat row = (-mat.row(0) + mat.row(1))/(double)spacing;
        row.copyTo(grad(cv::Rect(0,0,maxCols,1)));

        row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(double)spacing;
        row.copyTo(grad(cv::Rect(0,maxRows-1,maxCols,1)));

        /* centered elements */
        cv::Mat centeredMat = mat(cv::Rect(0,0,maxCols,maxRows-2));
        cv::Mat offsetMat = mat(cv::Rect(0,2,maxCols,maxRows-2));
        cv::Mat resultCenteredMat = (-centeredMat + offsetMat)/(((double)spacing)*2.0);

        resultCenteredMat.copyTo(grad(cv::Rect(0,1,maxCols, maxRows-2)));
        return grad;
    }


    cv::Mat HARRIS::smoothImage(const cv::Mat &image, const double sigma) {
        const int width = image.cols;
        const int height = image.rows;
        cv::Mat result(height, width, CV_64FC1);
        double *smoothed = (double *) result.data;

        vl_imsmooth_d(smoothed, width, (double *) image.data, width, height, width, sigma, sigma);

        return result;
    }
}
