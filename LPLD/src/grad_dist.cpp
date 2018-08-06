#include "grad_dist.hpp"
#include <algorithm>
#include <ctime>

namespace dip {
    Grad_Dist::Grad_Dist(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {
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

	void Grad_Dist::initCL() throw (std::runtime_error, cl::Error) {
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

				m_clSrcFileName = "grad_dist.cl";

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

    void Grad_Dist::compute() throw (std::runtime_error) {
        m_result = cv::Mat::zeros(m_ref_img.rows, m_ref_img.cols, CV_64FC1);
        
        m_test_img =  GPLib::get_luminance(m_test_img);
        m_ref_img =  GPLib::get_luminance(m_ref_img);

		std::vector<cv::Mat> T;
		std::vector<cv::Mat> R;

		std::vector<cv::Mat> T2;
		std::vector<cv::Mat> R2;

		initCL();

		vis_dist_GPU(m_test_img, T, 8, 8);
		vis_dist_GPU(m_ref_img, R, 8, 8);

		//vis_dist(m_test_img, T, 8, 8);
		//vis_dist(m_ref_img, R, 8, 8);
		//GPLib::writeCvMatToFile<double>(T[0], "matrixes/grad1.yml", true);
		//GPLib::writeCvMatToFile<double>(T[1], "matrixes/grad2.yml", true);
        cv::Mat d, m;
        for(unsigned int i = 0; i < T.size(); i++) {
            cv::pow(T[i] - R[i], 2.0, d);
            cv::max(T[i], R[i], m);
            m = m * (1/0.03) + 1;
        
            cv::divide(d, m, d);
            cv::resize(d, d, cv::Size(m_result.cols, m_result.rows));
            m_result = m_result + d;
        }
    }

	void Grad_Dist::vis_dist_GPU(cv::Mat &image, std::vector<cv::Mat> &dst, unsigned int rowBlock, unsigned int colBlock) {
		double median;
		cv::Mat result;
		cv::Mat gradients[2];
		cv::Scalar meanValue;
		std::vector<double> meanValuesVec;

		gradients[0] = gradientX(image, 1.0);
		gradients[1] = gradientY(image, 1.0);
		int sz_pad = static_cast<int>((gradients[0].rows / rowBlock));
		int sz_pad2 = static_cast<int>((gradients[0].cols / colBlock));
		if (gradients[0].rows % rowBlock != 0) {
			sz_pad++;
		}
		if (gradients[0].cols % colBlock != 0) {
			sz_pad2++;
		}
		int outSize = sz_pad *sz_pad2;
		int gradSize = gradients[0].rows * gradients[0].cols;

		cl::Buffer grad1_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, gradSize * sizeof(double), (double *)gradients[0].data);
		cl::Buffer grad2_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, gradSize * sizeof(double), (double *)gradients[1].data);
		cl::Buffer out1_Buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, outSize * sizeof(double));
		cl::Buffer out2_Buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, outSize * sizeof(double));

		try {
			std::ostringstream tmpStringStream;
			tmpStringStream << "-D COLBLOCK=" << colBlock << " " \
				<< "-D ROWBLOCK=" << rowBlock << " " \
				<< "-D ROWS=" << gradients[0].rows << " " \
				<< "-D COLS=" << gradients[0].cols;

			std::string compilerOptions = tmpStringStream.str();
			m_program.build(m_devicesVector, compilerOptions.c_str());

			cl::Kernel kernel = cl::Kernel(m_program, "grad_dist");


			kernel.setArg<cl::Buffer>(0, grad1_Buffer);
			kernel.setArg<cl::Buffer>(1, grad2_Buffer);
			kernel.setArg<cl::Buffer>(2, out1_Buffer);
			kernel.setArg<cl::Buffer>(3, out2_Buffer);

			cl::NDRange localSize = cl::NDRange(colBlock, rowBlock);
			cl::NDRange globalSize = cl::NDRange(sz_pad2 * colBlock, sz_pad * rowBlock);

			m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
			m_queue.finish();

			cv::Mat outMat(sz_pad, sz_pad2, CV_64FC1);
			m_queue.enqueueReadBuffer(out1_Buffer, CL_TRUE, 0, outSize * sizeof(double), outMat.data);
			cv::Mat outMat2(sz_pad, sz_pad2, CV_64FC1);
			m_queue.enqueueReadBuffer(out2_Buffer, CL_TRUE, 0, outSize * sizeof(double), outMat2.data);
			dst.push_back(outMat);
			dst.push_back(outMat2);

		}
		catch (cl::Error error) {
			if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devicesVector[0]) << std::endl;
			}
			throw;
		}
		return;



	}

    void Grad_Dist::vis_dist(cv::Mat &image, std::vector<cv::Mat> &dst, unsigned int rowBlock, unsigned int colBlock) {
        double median;
        cv::Mat result;
        cv::Mat gradients[2];
        cv::Scalar meanValue;
        std::vector<double> meanValuesVec;

        gradients[0] = gradientX(image, 1.0);
        gradients[1] = gradientY(image, 1.0);

        int sz_pad = static_cast<int>((gradients[0].rows / rowBlock));
        for(unsigned int i = 0; i < 2; i++) {
            for (int r = 0; r < gradients[i].rows; r += rowBlock) {
                for (int c = 0; c < gradients[i].cols; c += colBlock) {
                    cv::Mat tile = gradients[i](cv::Range(r, std::min(r + rowBlock, (unsigned int) gradients[i].rows)),
                                                cv::Range(c, std::min(c + colBlock, (unsigned int) gradients[i].cols)));

                    median = GPLib::getMedian<double>(tile);

                    for (int r = 0; r < tile.rows; r++) {
                        for (int c = 0; c < tile.cols; c++) {
                            tile.at<double>(r, c) = std::abs(tile.at<double>(r, c) - median);
                        }
                    }
                    meanValue = cv::mean(tile);
                    meanValuesVec.push_back(meanValue[0]);
                }
            }

            result = cv::Mat(1, meanValuesVec.size(), CV_64FC1);   
            memcpy(result.data,meanValuesVec.data(),meanValuesVec.size()*sizeof(double)); 
            result = result.reshape(1, sz_pad);
            
            dst.push_back(result);
            meanValuesVec.clear();
        }

        return;
    }


    cv::Mat Grad_Dist::gradientX(const cv::Mat & mat, double spacing) {
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


    cv::Mat Grad_Dist::gradientY(const cv::Mat & mat, double spacing) {
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


    cv::Mat Grad_Dist::im2col(const cv::Mat &src, unsigned int rowBlock, unsigned int colBlock) {
        int m = src.rows;
        int n = src.cols;

        // using right x = col; y = row
        int yB = m - rowBlock + 1;
        int xB = n - colBlock + 1;

        cv::Mat result = cv::Mat::zeros(xB*yB,rowBlock*colBlock,CV_64FC1);
        for(int i = 0; i< yB; i++)
        {
            for (int j = 0; j< xB; j++)
            {
                // here yours is in different order than I first thought:
                int rowIdx = j + i*xB;    // my intuition how to index the result
                //int rowIdx = i + j*yB;

                for(unsigned int yy =0; yy < rowBlock; ++yy)
                    for(unsigned int xx=0; xx < colBlock; ++xx)
                    {
                        // here take care of the transpose in the original method
                        //int colIdx = xx + yy*colBlock; // this would be not transposed
                        int colIdx = xx*rowBlock + yy; 

                        result.at<double>(rowIdx,colIdx) = src.at<double>(i+yy, j+xx);
                    }
            }
        }

        return result;
    }
}
