#include "mask_entropy_I.hpp"
#include <algorithm>
#include <ctime>

namespace dip {
    MASK_ENT_I::MASK_ENT_I(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {
        m_test_img =testImg;
        m_ref_img = referenceImg;

		m_platformNum = platformNum;
		m_deviceNum = deviceNum;
		m_deviceType = deviceType;

        if (m_ref_img.rows == 0 || m_ref_img.cols == 0)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows == 0 || m_ref_img.cols == 0)");
        if (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)
            throw std::runtime_error("ERROR: loadSources() - (m_ref_img.rows != m_test_img.rows || m_ref_img.cols != m_test_img.cols)");
    }


	void MASK_ENT_I::initCL() throw (std::runtime_error, cl::Error) {
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

				m_clSrcFileName = "mask_entropy_I.cl";

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


    void MASK_ENT_I::compute() throw (std::runtime_error) {
        m_result = cv::Mat::zeros(m_test_img.rows, m_test_img.cols, CV_64FC1);
        unsigned int windowSize = 3;

        m_test_img = GPLib::get_luminance(m_test_img);
        m_ref_img = GPLib::get_luminance(m_ref_img);

        cv::normalize(m_ref_img, m_ref_img, 0.0, 255.0, cv::NORM_MINMAX, CV_32SC1);
        cv::normalize(m_test_img, m_test_img, 0.0, 255.0, cv::NORM_MINMAX, CV_32SC1);

        cv::Mat refImgEntropy(m_ref_img.rows, m_ref_img.cols, CV_64FC1);
        cv::Mat testImgEntropy(m_test_img.rows, m_test_img.cols, CV_64FC1);

		initCL();

		computeEntropy_GPU(m_ref_img, refImgEntropy, windowSize);
		computeEntropy_GPU(cv::abs(m_test_img - m_ref_img), testImgEntropy, windowSize);

        //computeEntropy(m_ref_img, refImgEntropy, windowSize);
        //computeEntropy(cv::abs(m_test_img - m_ref_img), testImgEntropy, windowSize);
		
        double *refImgEntrPtr = (double *) refImgEntropy.data;
        double *testImgEntrPtr = (double *) testImgEntropy.data;
        double *resultPtr = (double *) m_result.data;

        for (int i = 0; i < m_result.rows * m_result.cols; i++) {
            if (*refImgEntrPtr != 0.0) {
                *resultPtr = *testImgEntrPtr / *refImgEntrPtr;
            }

            resultPtr++;
            testImgEntrPtr++;
            refImgEntrPtr++;
        }
    }

	void MASK_ENT_I::computeEntropy_GPU(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize) {
		cv::Mat padded_srcImg;
		double sum = 0.0;
		float histogramVals[256] = { 0.0 };

		int cycles = image.rows * image.cols;
		int cols = image.cols;
		int rows = image.rows;

		cv::copyMakeBorder(image, padded_srcImg, 1, 1, 1, 1, cv::BORDER_CONSTANT);

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

			cl::Kernel kernel = cl::Kernel(m_program, "mask_entropy");


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

    void MASK_ENT_I::computeEntropy(const cv::Mat &image, cv::Mat &dst, unsigned int windowSize) {
        cv::Mat padded_srcImg;
        double sum = 0.0;
        float histogramVals[256] = {0.0};
        
        cv::copyMakeBorder(image, padded_srcImg, 1, 1, 1, 1, cv::BORDER_CONSTANT);

        for (int r = 1; r < image.rows + 1; r++) {
            for (int c = 1; c < image.cols + 1; c++) {
                cv::Mat tile = padded_srcImg(cv::Rect(c-1, r-1, 3, 3));

                for (int r_roi = 0; r_roi < 3; r_roi++) {
                    for (int c_roi = 0; c_roi < 3; c_roi++) {
                        histogramVals[tile.at<int>(r_roi, c_roi)] += 1;
                    }
                }

                for (int i = 0; i < 256; i++) {
                    if (histogramVals[i] != 0)
                        sum += (histogramVals[i] / 9.0) * (std::log((histogramVals[i] / 9.0)) / std::log(2.0));
                }

                dst.at<double>(r-1, c-1) =  -1.0 * sum;

                for (int i = 0; i < 256; i++)
                    histogramVals[i] = 0;

                sum = 0.0;
            }
        }

        return;
    }
}