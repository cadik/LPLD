#include "spyrDist.hpp"
#include <ctime>

namespace dip {
    SPYR_DIST::SPYR_DIST(cv::Mat &testImg, cv::Mat &referenceImg) throw (std::runtime_error) {
        m_test_img = testImg;
        m_ref_img = referenceImg;
		cv::Mat m_result;

        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

    }

    void SPYR_DIST::compute() throw (std::runtime_error) {
        m_ref_img = GPLib::get_luminance(m_ref_img);
        m_test_img = GPLib::get_luminance(m_test_img);

        m_ref_img.convertTo(m_ref_img, CV_32FC1);
        m_test_img.convertTo(m_test_img, CV_32FC1);

        std::vector<cv::Mat> T = vis_dis(m_test_img);
        std::vector<cv::Mat> R = vis_dis(m_ref_img);

        m_result = cv::Mat::zeros(m_ref_img.rows, m_ref_img.cols, CV_64FC1);

        for(size_t i = 0; i < T.size(); i++) {
            cv::Mat d = T[i] - R[i];
            d =  d.mul(d);
            cv::resize(d, d, cv::Size(m_result.cols, m_result.rows), 0.0, 0.0, cv::INTER_LINEAR);
            m_result = m_result + d;
        }
    }


    std::vector<cv::Mat> SPYR_DIST::vis_dis(const cv::Mat &inputMat) {
        std::vector<cv::Mat> result;

        PFILTER PF;
        MATRIX P_Matrix = NewMatrix(inputMat.rows, inputMat.cols);

        PF = LoadPFilter("sp3Filters");
        if (PF == NULL)
            throw std::runtime_error("ERROR: steerpyr_filter file not found!");

        P_Matrix->rows = inputMat.rows;
        P_Matrix->columns = inputMat.cols;

        float *pPtr = (float *) inputMat.data;
        float *p_matPtr = P_Matrix->values;

        for (int i = 0; i < inputMat.rows * inputMat.cols; i++) {
            *p_matPtr = *pPtr;
            p_matPtr++; pPtr++;
        }
                
        int levels = 4;

        PYRAMID pyramid = CreatePyramid(P_Matrix, PF, levels);
        cv::Mat tmpMat = get_band(pyramid, 0, 0);

        result.push_back(computeSTD(tmpMat));

        for(int i = 1; i < pyramid->num_levels + 1; i++) {
            for(int j = 0; j < PF->num_orientations; j++) {
                tmpMat = get_band(pyramid, i, j);

                result.push_back(computeSTD(tmpMat));
            }
        }

        tmpMat = get_band(pyramid, pyramid->num_levels + 1, 0);
        result.push_back(computeSTD(tmpMat));

        return result;
    }


    cv::Mat SPYR_DIST::get_band(PYRAMID pyr, int band, int orienation) {
        MATRIX tmpMatrix;
        float minOne = 1.0f;

        if (band == 0)
            tmpMatrix = pyr->hiband;
        else if (band == pyr->num_levels + 1)
            tmpMatrix = pyr->lowband;
        else {
            tmpMatrix = GetSubbandImage(pyr, band - 1, orienation);
            minOne = -1.0f;
        }

        cv::Mat result(tmpMatrix->rows, tmpMatrix->columns, CV_32FC1);

        float *resPtr = (float *) result.data;
        float *tmpPtr = tmpMatrix->values;

        for (int i = 0; i < tmpMatrix->rows * tmpMatrix->columns; i++) {
            *resPtr++ = minOne * (*tmpPtr++);
        }

        result.convertTo(result, CV_64FC1);

        return result;
    }

    cv::Mat SPYR_DIST::computeSTD(cv::Mat &inputMat) {
        size_t windowSize = 8;
        
        unsigned int padd_w = roundUp(inputMat.cols, windowSize);
        unsigned int padd_h = roundUp(inputMat.rows, windowSize);

        cv::copyMakeBorder(inputMat, inputMat, 0, padd_h - inputMat.rows, 0, \
                            padd_w - inputMat.cols, cv::BORDER_CONSTANT, cv::Scalar(0));

        unsigned int res_w = (unsigned int) (inputMat.cols / windowSize);
        unsigned int res_h = (unsigned int) (inputMat.rows / windowSize);
                
        cv::Mat sResult(res_h, res_w, CV_64FC1);
        int res_r = 0, res_c = 0;

        for(unsigned int c = 0; c <= (inputMat.cols - windowSize); c += windowSize ) {
            res_r = 0;
            for(unsigned int r = 0; r <= (inputMat.rows - windowSize); r += windowSize ) {
                cv::Mat roi = inputMat(cv::Rect(c, r, windowSize, windowSize));

                cv::Scalar mean,stddev;
                cv::meanStdDev(roi, mean, stddev);
                sResult.at<double>(res_r, res_c) = stddev[0];

                res_r++;
            }
            res_c++;
        }

        return sResult;
    }


    unsigned int SPYR_DIST::roundUp(unsigned int value, unsigned int multiple) const {
        // Determine how far past the nearest multiple the value is
        unsigned int remainder = value % multiple;

        // Add the difference to make the value a multiple
        if(remainder != 0)
            value += (multiple-remainder);
        return value;
    }
}
