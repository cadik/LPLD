#ifndef DIP_HDRVDP_HPP
#define DIP_HDRVDP_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <stdexcept>
//#include <iostream>

#include <CL/cl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "stdafx.h"
#include "CLSharedLib.hpp"
#include "GeneralPurposeLib.hpp"
#include "metric_params.hpp"
#include "hdrvdpVisPathway.hpp"
#include "csvIterator.hpp"
#include "hdrvdpHelper.hpp"
#include "hdrvdpVisualize.hpp"
#include "clClass.hpp"

namespace dip {
    class HDRVDP {
    public:
        // Constructors
        HDRVDP() {
            try {
                clObj = new clClass();
            } catch(...) {
                throw;
            }
        };

        HDRVDP(int platformNum, cl_device_type deviceType, int deviceNum) {
            try {
                clObj = new clClass(platformNum, deviceType, deviceNum);

            } catch(...) {
                throw;
            }
        };

        void compute(std::string reference_img, std::string test_img, COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error);
        void compute(cv::Mat reference_img, cv::Mat test_img, COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error);
        const cv::Mat getTestImage() const {return m_testBackup; }
        const cv::Mat getP_Map() const {return P_map; }
        std::vector<std::vector<cv::Mat>> getComputedBands(cv::Mat reference_img, cv::Mat test_img, COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error);

    private:
        void loadImages(std::string referenceName, std::string testName, COLOR_ENC color_encoding) throw (std::runtime_error);
        void runComputation(COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error);
        void load_spectral_resp() throw (std::runtime_error);
        void setSources() throw (std::runtime_error, cl::Error);
        void display_mode_Luma();
        void display_mode_sRGB() throw (cl::Error);
        
        cv::Mat mutual_masking(cv::Mat &testBand, cv::Mat &refBand);
        cv::Mat get_band(PYRAMID pyr, int band, int orienation);
        void set_band(PYRAMID pyr, int band, int orienation, const cv::Mat &newBand);
        void copyPyramid(const PYRAMID src, PYRAMID dst);
        //cv::Mat get_band_size();

        clClass *clObj;

        MetricParams m_metricPar;
        HDRVDP_VisPathway *m_visualPathway_refImg;
        HDRVDP_VisPathway *m_visualPathway_testImg;

        //std::string m_ref_img_path;
        //std::string m_test_img_path;
        COLOR_ENC m_color_encoding;

        cv::Mat m_refMat;
        cv::Mat m_testMat;
        cv::Mat m_testBackup;

        cv::Mat S_map;
        cv::Mat C_map;
        double C_max;
        double Q;
        double Q_MOS;
        cv::Mat P_map;
        double P_det;

        //std::vector


        int cvMatType;

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
