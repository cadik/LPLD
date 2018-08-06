#include "mask_entropy_multi.hpp"
#include <algorithm>
#include <ctime>

namespace dip {
    MASK_ENT_MULTI::MASK_ENT_MULTI(cv::Mat &testImg, cv::Mat &referenceImg, int level, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {
        m_level = level;

        m_test_img = testImg;
        m_ref_img = referenceImg;


		m_platformNum = platformNum;
		m_deviceNum = deviceNum;
		m_deviceType = deviceType;

        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");

    }
	void MASK_ENT_MULTI::initCL() throw (std::runtime_error, cl::Error) {
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

				m_clSrcFileName = "mask_entropy_multi.cl";

#ifdef _MSC_VER
				std::ifstream sourceFile("src\\" + m_clSrcFileName);
#else
				std::ifstream sourceFile("cl/" + m_clSrcFileName);
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


	void MASK_ENT_MULTI::compute() throw (std::runtime_error) {
		unsigned int windowSize = 5;
		unsigned int levels = 5;

		m_test_img = GPLib::get_luminance(m_test_img);
		m_ref_img = GPLib::get_luminance(m_ref_img);

		cv::normalize(m_ref_img, m_ref_img, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
		cv::normalize(m_test_img, m_test_img, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);

		std::vector<cv::Mat> pyr_test(6);
        std::vector<cv::Mat> pyr_ref(6);
        cv::Mat D;
		initCL();

        cv::buildPyramid(m_test_img, pyr_test, levels, cv::BORDER_DEFAULT);
        cv::buildPyramid(m_ref_img, pyr_ref, levels, cv::BORDER_DEFAULT);

        cv::Mat refImgEntropy;//(m_ref_img.rows, m_ref_img.cols, CV_64FC1);
        cv::Mat testImgEntropy;//(m_test_img.rows, m_test_img.cols, CV_64FC1);
        cv::Mat ref_e, test_e;

        for (unsigned int pyrNum = 0; pyrNum < levels; pyrNum++) {
            double scale_part = std::pow(2.0, pyrNum);
            double scaleA = 1.0 / scale_part;

            cv::resize(pyr_ref[pyrNum], refImgEntropy, cv::Size(0 ,0), scaleA, scaleA, cv::INTER_CUBIC); 
            cv::resize(cv::abs(pyr_test[pyrNum] - pyr_ref[pyrNum]), testImgEntropy, cv::Size(0 ,0), scaleA, scaleA, cv::INTER_CUBIC);

            computeEntropy_GPU(refImgEntropy, ref_e, windowSize);
            computeEntropy_GPU(testImgEntropy, test_e, windowSize);

            cv::divide(test_e, ref_e, D);
            cv::resize(D, D, cv::Size(0, 0), scale_part, scale_part, cv::INTER_CUBIC); 
            
            m_result.push_back(D);
        }
    }


	void MASK_ENT_MULTI::computeEntropy_GPU(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize) {
		cv::Mat padded_srcImg;
		double sum = 0.0;
		float histogramVals[256] = { 0.0 };

		int cycles = image.rows * image.cols;
		int cols = image.cols;
		int rows = image.rows;
		size_t half_window = (int)(windowSize / 2);

		dst = cv::Mat::zeros(image.size(), CV_64FC1);

		cv::copyMakeBorder(image, padded_srcImg, half_window, half_window, half_window, half_window, cv::BORDER_CONSTANT);

		int sizeofImage = padded_srcImg.rows * padded_srcImg.cols;
		padded_srcImg.convertTo(padded_srcImg, CV_64FC1);

		cl::Buffer image_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeofImage * sizeof(double), (double *)padded_srcImg.data);
		cl::Buffer out_Buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, cycles * sizeof(double));

		try {
			std::ostringstream tmpStringStream;
			tmpStringStream << "-D WINDOWSIZE=" << windowSize << " " \
				<< "-D PADDED_ROWS=" << padded_srcImg.rows << " " \
				<< "-D PADDED_COLS=" << padded_srcImg.cols << " " \
				<< "-D ROWS=" << image.rows << " " \
				<< "-D COLS=" << image.cols;

			std::string compilerOptions = tmpStringStream.str();
			m_program.build(m_devicesVector, compilerOptions.c_str());

			cl::Kernel kernel = cl::Kernel(m_program, "mask_entropy_multi");


			kernel.setArg<cl::Buffer>(0, image_Buffer);
			kernel.setArg<cl::Buffer>(1, out_Buffer);

			cl::NDRange globalSize = cl::NDRange(cols, rows);

			m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
			m_queue.finish();

			//cv::Mat outMat(rows, cols, CV_64FC1);
			m_queue.enqueueReadBuffer(out_Buffer, CL_TRUE, 0, cycles * sizeof(double), dst.data);
			//GPLib::writeCvMatToFile<double>(dst, "matrixes/matrix.yml", true);

		}
		catch (cl::Error error) {
			if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devicesVector[0]) << std::endl;
			}
			throw;
		}
	}

    void MASK_ENT_MULTI::computeEntropy(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize) {
        cv::Mat padded_srcImg;
        double sum = 0.0;
        float histogramVals[256] = {0.0};

        size_t half_window = (int) (windowSize / 2);
        
        dst = cv::Mat::zeros(image.size(), CV_64FC1);
        
        cv::copyMakeBorder(image, padded_srcImg, half_window, half_window, half_window, half_window, cv::BORDER_CONSTANT);

        for (unsigned int r = half_window; r < (image.rows + half_window); r++) {
            for (unsigned int c = half_window; c < (image.cols + half_window); c++) {
                cv::Mat tile = padded_srcImg(cv::Rect(c-half_window, r-half_window, windowSize, windowSize));

                for (unsigned int i = 0; i < windowSize*windowSize; i++) {
                        histogramVals[*((unsigned char *) tile.data + i)] += 1;
                }

                for (int i = 0; i < 256; i++) {
                    if (histogramVals[i] != 0)
                        sum += (histogramVals[i] / (windowSize*windowSize)) * (std::log((histogramVals[i] / (windowSize*windowSize))) / std::log(2.0));
                }

                dst.at<double>(r-half_window, c-half_window) =  -1.0 * sum;

                for (int i = 0; i < 256; i++)
                    histogramVals[i] = 0;

                sum = 0.0;
            }
        }
        return;
    }
}