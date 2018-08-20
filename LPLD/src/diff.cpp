#include "diff.hpp"
#include <ctime>

namespace dip {
    DIFF::DIFF(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {
		m_platformNum = platformNum;
		m_deviceNum = deviceNum;
		m_deviceType = deviceType;

        m_test_img = testImg;
        m_ref_img = referenceImg;

        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

    }

    void DIFF::compute() throw (std::runtime_error) {
        m_result = cv::Mat(m_ref_img.size(), CV_64FC1);
		
        m_ref_img = GPLib::get_luminance(m_ref_img);
        m_test_img = GPLib::get_luminance(m_test_img);

        //size_t numOfPixels = m_result.rows * m_result.cols;
		
		//cv::subtract(m_test_img, m_ref_img, m_result);
		/*for (unsigned int i = 0; i < numOfPixels; i++) {
			*((double*)m_result.data + i) = abs(*((double*)m_test_img.data + i) - *((double*)m_ref_img.data + i));
		}*/
		m_result = m_test_img - m_ref_img;
    }

	template<typename T> 
	void private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error) {


	}

	void DIFF::initCL() throw (std::runtime_error, cl::Error) {
		std::vector<cl::Platform> platformsList;

		try {
			cl::Platform::get(&platformsList);

			try {
				m_platform = platformsList.at(m_platformNum);
			}
			catch (const std::out_of_range& oor) {
				std::cerr << "Out of Range error (std::vector<cl::Platform>)!\n" << \
					"Please choose correct cl::Platform number (--cl-platform param)!\n";
				throw oor;
			}

			//             dip::CLInfo::printQuickPlatformInfo(m_platform, m_platformNum);

			cl_context_properties cprops[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)m_platform(), 0 };
			m_context = cl::Context(m_deviceType, cprops);

			std::vector<cl::Device> tmp_devicesList = m_context.getInfo<CL_CONTEXT_DEVICES>();

			try {
				m_device = tmp_devicesList.at(m_deviceNum);
				m_dev_doubleSupport = dip::CLInfo::hasDoublePrecision(m_device);

				m_clSrcFileName = "diff.cl";

#ifdef _MSC_VER
				std::ifstream sourceFile("src\\" + m_clSrcFileName);
#else
				std::ifstream sourceFile("src/" + m_clSrcFileName);
#endif

				std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), \
					(std::istreambuf_iterator<char>()));

				cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
				m_program = cl::Program(m_context, source);

				m_devicesVector.push_back(m_device);
			}
			catch (const std::out_of_range& oor) {
				std::cerr << "Out of Range error (std::vector<cl::Device>)!\n" << \
					"Please choose correct cl::Device number (--cl-device param)!\n";
				throw oor;
			}

			//             dip::CLInfo::printQuickDeviceInfo(m_device, m_deviceNum);

#ifdef PROFILE
			m_queue = cl::CommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE);
#else
			m_queue = cl::CommandQueue(m_context, m_device);
#endif
		}
		catch (...) {
			throw;
		}
	}
	
}
