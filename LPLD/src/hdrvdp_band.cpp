#include "hdrvdp_band.hpp"
#include <algorithm>
#include <ctime>

namespace dip {
	HDRVDP_BAND::HDRVDP_BAND(cv::Mat &testImg, cv::Mat &referenceImg, int level) throw (std::runtime_error) {
		m_level = level;

		m_test_img = testImg.clone();
		m_ref_img = referenceImg.clone();


		if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
			throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
		if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
			throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

		m_ref_img.convertTo(m_ref_img, CV_32FC3);
		m_test_img.convertTo(m_test_img, CV_32FC3);

		// normalize to 0-1
		for (int i = 0; i < m_test_img.cols * m_test_img.rows * m_test_img.channels(); i++) {
			*((float *)m_test_img.data + i) = *((float *)m_test_img.data + i) / 255.0f;
			*((float *)m_ref_img.data + i) = *((float *)m_ref_img.data + i) / 255.0f;
		}

	}

	HDRVDP_BAND::HDRVDP_BAND(cv::Mat &testImg, cv::Mat &referenceImg, int level, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {
		m_level = level;

		m_test_img = testImg.clone();
		m_ref_img = referenceImg.clone();

		m_platformNum = platformNum;
		m_deviceNum = deviceNum;
		m_deviceType = deviceType;

		if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
			throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
		if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
			throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

		m_ref_img.convertTo(m_ref_img, CV_32FC3);
		m_test_img.convertTo(m_test_img, CV_32FC3);

		// normalize to 0-1
		for (int i = 0; i < m_test_img.cols * m_test_img.rows * m_test_img.channels(); i++) {
			*((float *)m_test_img.data + i) = *((float *)m_test_img.data + i) / 255.0f;
			*((float *)m_ref_img.data + i) = *((float *)m_ref_img.data + i) / 255.0f;
		}

	}


    void HDRVDP_BAND::compute() throw (std::runtime_error) {
        cv::Mat test_L = 0.2f + (99.8f * m_test_img);
        cv::Mat reference_L = 0.2f + (99.8f * m_ref_img);
		
        dip::HDRVDP hdrvpObj(m_platformNum, m_deviceType, m_deviceNum);

        double ppd = 60.0;
		std::vector<std::vector<cv::Mat>>  band_diff;
        //hdrvpObj.compute(reference_L, test_L, dip::COLOR_ENC::RGB_BT_709, ppd);
		band_diff = hdrvpObj.getComputedBands(reference_L, test_L, dip::COLOR_ENC::RGB_BT_709, ppd);

        for(size_t b = 0; b < band_diff.size(); b++) {
            cv::Mat D = cv::Mat::zeros(m_test_img.rows, m_test_img.cols, CV_64FC1);

            for(size_t o = 0; o < band_diff[b].size(); o++) {
               //GPLib::writeCvMatToFile<double>(result[0], "result.yml", true, false);
                cv::resize(cv::abs(band_diff[b][o]), band_diff[b][o], D.size(), 0.0, 0.0, CV_INTER_LINEAR);
                D = D + band_diff[b][o];
            }
            m_result.push_back(D);
        }
    }


}