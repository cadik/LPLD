#include "hog9.hpp"
#include <ctime>

namespace dip {
    HOG9::HOG9(cv::Mat &testImg, cv::Mat &referenceImg) throw (std::runtime_error) {
        m_test_img = testImg;
        m_ref_img = referenceImg;


        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

    }


    void HOG9::compute() throw (std::runtime_error) {
        cv::Mat hog_t, hog_r;
        cv::Mat powResult;
        std::vector<cv::Mat> powResChannels(50);

        hog_t = get_hogdescriptor(m_test_img);
        hog_r = get_hogdescriptor(m_ref_img);

        cv::pow(hog_t - hog_r, 2.0, powResult);

        cv::split(powResult, powResChannels);

        unsigned int powResWidth = powResult.cols;
        unsigned int powResHeight = powResult.rows;
        cv::Mat tmpResult = cv::Mat::zeros(powResHeight, powResWidth, CV_64FC1);
        double *tmpResPtr = (double *) tmpResult.data;
        float *powResChanPtr = NULL;

        for(unsigned int j = 0; j < powResChannels.size(); j++) {
            powResChanPtr = (float *) powResChannels[j].data;

            for(unsigned int i = 0; i < powResWidth * powResHeight; i++) {
                *tmpResPtr += *powResChanPtr++;
                tmpResPtr++;
            }

            tmpResPtr = (double *) tmpResult.data;
        }
        
        tmpResPtr = (double *) tmpResult.data;
        for(unsigned int i = 0; i < powResWidth * powResHeight; i++) {
            *tmpResPtr = *tmpResPtr / powResChannels.size();
            tmpResPtr++;
        }


        cv::resize(tmpResult, m_result, cv::Size(m_ref_img.cols, m_ref_img.rows), 0.0, 0.0, CV_INTER_CUBIC);
        //char buffer[100];
        //snprintf(buffer, sizeof(buffer), "hog9_t_%d.yml", 0);//i+1);
        //GPLib::writeCvMatToFile<double>(m_result, buffer, false, true);
    }

cv::Mat HOG9::get_hogdescriptor(const cv::Mat &inputMat)
    {   
        cv::Mat result, tmpMat;
        std::vector<cv::Mat> tmpChannelsVec(3);
		std::vector<cv::Mat> hogMatrix(0);
        unsigned int k = 0;
        unsigned int inputWidth = inputMat.cols;
        unsigned int inputHeight = inputMat.rows;
        unsigned int numOfChannels = inputMat.channels();

        inputMat.convertTo(tmpMat, CV_32FC(numOfChannels));
        cv::cvtColor(tmpMat, tmpMat, CV_BGR2RGB);

        float* imgPtr = (float*)malloc(inputWidth * inputHeight * numOfChannels * sizeof(float));
        cv::split(tmpMat, tmpChannelsVec);

        for(unsigned int i = 0; i < tmpChannelsVec.size(); i++) {
            float *chanPtr = (float *) tmpChannelsVec[i].data;

            for(unsigned int j = 0; j < inputWidth * inputHeight; j++) {
                imgPtr[k] = *chanPtr++;
                k++;
            }
        }

        vl_size numOrientations = 9;
        vl_size numChannels = numOfChannels;
        vl_size cellSize = 8;

        VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE) ;
        vl_hog_put_image(hog, imgPtr, inputWidth, inputHeight, numChannels, cellSize);

        vl_size hogWidth = vl_hog_get_width(hog) ;
        vl_size hogHeight = vl_hog_get_height(hog) ;
        vl_size hogDimension = vl_hog_get_dimension(hog) ;
		
        float *hogArray = (float*) vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
        vl_hog_extract(hog, hogArray) ;
        vl_hog_delete(hog);

        for (unsigned int i = 0; i < hogDimension; i++)
            hogMatrix.push_back(cv::Mat(hogHeight, hogWidth, CV_32FC1, hogArray + (i*hogWidth*hogHeight)));
        
        cv::merge(hogMatrix, result);

        free(imgPtr);
        vl_free(hogArray);

        return result;
    }
}