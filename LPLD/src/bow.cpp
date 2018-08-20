#include "bow.hpp"
#include <ctime>

namespace dip {
    void BOW::compute() throw (std::runtime_error) {

		try {
			initCL();
			if (m_dev_doubleSupport) {
				private_compute<double>(CV_64FC1);
				//standard_compute();
			}
			else
				private_compute<float>(CV_32FC1);
		}
		catch (...) {
			throw;
		}


    }

	void BOW::standard_compute() throw (std::runtime_error){

		m_lumTestImg = GPLib::get_luminance(m_test_img);
		m_lumRefImg = GPLib::get_luminance(m_ref_img);

		m_diff = m_lumTestImg - m_lumRefImg;



		double maxLen = static_cast<double>(m_diff.rows >= m_diff.cols ? m_diff.rows : m_diff.cols);
		double downSampleRatio = 0.0;
		if (maxLen > m_params.getCanonicalImgRes())
			downSampleRatio = m_params.getCanonicalImgRes() / maxLen;
		else
			downSampleRatio = maxLen / m_params.getCanonicalImgRes();

		cv::resize(m_diff, m_diff, cv::Size(), downSampleRatio, downSampleRatio, CV_INTER_LINEAR);


		unsigned int stride = m_params.getStride();
		unsigned int nRows = (unsigned int)(m_diff.rows / stride);
		unsigned int nCols = (unsigned int)(m_diff.cols / stride);

		cv::Mat artifactClusterIndices = cv::Mat::zeros(nRows, nCols, CV_32FC1);
		double minVal, maxVal;

		cv::Mat dictionary = m_params.getDictionary();
		cv::transpose(dictionary, dictionary);
		cv::Point minLoc;

		for (unsigned int y = 0; y < nRows; y++) {
			for (unsigned int x = 0; x < nCols; x++) {
				unsigned int patchCord_X = x * stride + 1;
				unsigned int patchCord_Y = y * stride + 1;

				std::vector<double> patchVector = computeDCT_Decsriptor(patchCord_X, patchCord_Y);
				cv::Mat patchDescriptor(patchVector);

				patchDescriptor = cv::repeat(patchDescriptor, 1, m_params.getDictionarySize());

				cv::Mat tmp = cv::abs(patchDescriptor - dictionary);
				cv::Mat dist = cv::Mat::zeros(1, tmp.cols, CV_64FC1);
				for (int c = 0; c < tmp.cols; c++) {
					for (int r = 0; r < tmp.rows; r++) {
						dist.at<double>(0, c) += tmp.at<double>(r, c);
					}
				}

				cv::minMaxLoc(dist, &minVal, &maxVal, &minLoc);
				artifactClusterIndices.at<int>(y, x) = static_cast<int>(minLoc.x);
			}
		}


		int nClusters = m_params.getDictionarySize();
		int neighborRadius = std::max((int)(m_params.getPatchSize() / 2), 16);
		//cv::Mat H(nRows, nCols, CV_64FC(nClusters));

		H_vec = std::vector<cv::Mat>(32);//reserve(nClusters)

		for (std::vector<cv::Mat>::iterator it = H_vec.begin(); it != H_vec.end(); ++it)
			(*it) = cv::Mat::zeros(nRows, nCols, CV_64FC1);

		cv::Mat gaussianKernel = cv::Mat(1, neighborRadius * 2 + 1, CV_64FC1);
		double *gauss_Ptr = (double *)gaussianKernel.data;

		for (int i = -neighborRadius; i <= neighborRadius; i++) {
			*gauss_Ptr++ = std::exp((-(i*i)) / std::pow(0.5 * neighborRadius, 2.0));
		}

		gaussianKernel = gaussianKernel.t() * gaussianKernel;

		cv::Mat expandedImg = cv::Mat::zeros(2 * neighborRadius + nRows, 2 * neighborRadius + nCols, CV_32FC1);
		artifactClusterIndices.copyTo(expandedImg(cv::Rect(neighborRadius, neighborRadius, nCols, nRows)));

		for (unsigned int y = 0; y < nRows; y++) {
			for (unsigned int x = 0; x < nCols; x++) {
				cv::Mat tmpROI = expandedImg(cv::Rect(x, y, 2 * neighborRadius + 1, 2 * neighborRadius + 1));

				for (int roiX = 0; roiX < tmpROI.cols; roiX++) {
					for (int roiY = 0; roiY < tmpROI.rows; roiY++) {
						cv::Mat tmpHeader = H_vec[tmpROI.at<int>(roiY, roiX)];
						double tmp = tmpROI.at<int>(roiY, roiX);
						tmpHeader.at<double>(y, x) += gaussianKernel.at<double>(roiY, roiX);
					}
				}
			}
		}

		for (unsigned int i = 0; i < H_vec.size(); i++)
			cv::resize(H_vec[i], H_vec[i], m_diff.size(), 0.0, 0.0, CV_INTER_LINEAR);
		/*for (int i = 0; i < 32; i++) {
			cv::imshow("test", H_vec[i]);
			cv::waitKey();
		}*/
		//GPLib::writeCvMatToFile<double>(H_vec[0], "gaussianKernel.yml", true);
	}

    std::vector<double> BOW::computeDCT_Decsriptor(int patchCord_X, int patchCord_Y, \
                                                   double randomRotation) {
        std::vector<double> descriptor(0);
        int patchSize = m_params.getPatchSize();
        cv::Mat subImage(patchSize, patchSize, CV_64FC1);
        int halfPathcSize = static_cast<int>(patchSize / 2);
        int top_x, top_y;

        if (randomRotation != 0.0) {
            // !TODO
        }
        else {
            int pixCordTmp_X = static_cast<int>(patchCord_X - std::max(0, patchCord_X + halfPathcSize - (m_diff.cols - 1)));
            int pixCordTmp_Y = static_cast<int>(patchCord_Y - std::max(0, patchCord_Y + halfPathcSize - (m_diff.rows - 1)));
            pixCordTmp_X = pixCordTmp_X + std::max(0, -1*(pixCordTmp_X - halfPathcSize));
            pixCordTmp_Y = pixCordTmp_Y + std::max(0, -1*(pixCordTmp_Y - halfPathcSize));

            top_x = pixCordTmp_X - (halfPathcSize - 1);
            top_y = pixCordTmp_Y - (halfPathcSize - 1);
        }

        cv::Rect subImgRoi(top_x, top_y, patchSize, patchSize);
        cv::Mat roi = m_diff(subImgRoi);
        roi.copyTo(subImage);

       
        //double *roi_Ptr = (double *) roi.data;

        //for(int x = 0; x < halfPathcSize; x++) {
        //    for(int y = 0; y < halfPathcSize; x++) {

        //    }
        //}

        //std::cout << "pX: " << patchCord_X << "    pY: " << patchCord_Y << "  ->   " << subImgRoi << std::endl;
            

        // !deep copy
        //subImage = roi.clone();

        //// !check performance!!
        unsigned int descriptorDim = m_params.getPatchSize();
        cv::Scalar meanVal;

		
        if (m_params.lumInvariantPatch()) {
            meanVal = cv::mean(subImage);
            subImage -= meanVal;
        }

        if (m_params.contrastInvariantPatch()) {
            double subImageVar;
            if (m_params.lumInvariantPatch())
                subImageVar = std::sqrt(std::max(0.005, getVariance(subImage, meanVal[0])));
            else {
                meanVal = cv::mean(subImage);
                subImageVar = std::sqrt(std::max(0.005, getVariance(subImage, meanVal[0])));
            }

            subImage *= (1.0 /subImageVar);
        }
		
        cv::Mat dctSubImage;
        cv::dct(subImage, dctSubImage);
		
        int dctOffset = 0;
        if(m_params.lumInvariantPatch()) {
            dctOffset = 1;
            if (descriptorDim <= 4)
                descriptorDim = descriptorDim - 1;
        }

        std::vector<std::pair<int, int>> dctZigZagOrder = m_params.getDCTZigZagOrder();
        for(unsigned int dctCoeff = 0; dctCoeff < descriptorDim; dctCoeff++) {
            unsigned int x = dctZigZagOrder[dctCoeff + dctOffset].first;
            unsigned int y = dctZigZagOrder[dctCoeff + dctOffset].second;

            descriptor.push_back(dctSubImage.at<double>(x,y));
        }
		
        return descriptor;
    }

    double BOW::getVariance(cv::Mat &inputMat, double meanVal) {
        size_t numOfItems = inputMat.rows * inputMat.cols;
        double *input_Ptr = (double *) inputMat.data;

        double temp = 0;
        for(unsigned int i = 0; i < numOfItems; i++) {
            temp += (meanVal-(*input_Ptr))*(meanVal-(*input_Ptr));
            input_Ptr++;
        }

        return temp/numOfItems;
    }

	void BOW::initCL() throw (std::runtime_error, cl::Error) {
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

				m_clSrcFileName = "bow.cl";

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

	template void BOW::private_compute<double>(int cv_mat_type) throw (std::runtime_error, cl::Error);
	template void BOW::private_compute<float>(int cv_mat_type) throw (std::runtime_error, cl::Error);
	
	template <typename T>
	void BOW::private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error) {

		m_lumTestImg = GPLib::get_luminance(m_test_img);
		m_lumRefImg = GPLib::get_luminance(m_ref_img);

		m_lumTestImg.convertTo(m_lumTestImg, cv_mat_type);
		m_lumRefImg.convertTo(m_lumRefImg, cv_mat_type);

		m_diff = m_lumTestImg - m_lumRefImg;
		//m_diff.convertTo(m_diff, cv_mat_type);

		T maxLen = static_cast<T>(m_diff.rows >= m_diff.cols ? m_diff.rows : m_diff.cols);
		/**T downSampleRatio = 0.0;
		if (maxLen > m_params.getCanonicalImgRes())
			downSampleRatio = m_params.getCanonicalImgRes() / maxLen;
		else
			downSampleRatio = maxLen / m_params.getCanonicalImgRes();

		cv::resize(m_diff, m_diff, cv::Size(), downSampleRatio, downSampleRatio, CV_INTER_LINEAR);**/


		unsigned int stride = m_params.getStride();
		unsigned int nRows = (unsigned int)(m_diff.rows / stride);
		unsigned int nCols = (unsigned int)(m_diff.cols / stride);
		int neighborRadius = std::max((int)(m_params.getPatchSize() / 2), 16);

		cv::Mat artifactClusterIndices = cv::Mat::zeros(nRows, nCols, cv_mat_type);
		double minVal, maxVal;

		cv::Mat dictionary = m_params.getDictionary();
		
		cv::transpose(dictionary, dictionary);
		dictionary.convertTo(dictionary, cv_mat_type);
		cv::Point minLoc;

		int globalCycles = nRows * nCols;

		int M = m_diff.rows;
		int N = m_diff.cols;

		size_t deviceDataSize = M * N;
		std::vector<std::pair<int, int>> tat = m_params.getDCTZigZagOrder();

		int *zz_order = (int *)calloc(512, sizeof(int));
		size_t zz_size = 512 * sizeof(int);
		for (int i = 0; i < tat.size(); i++) {
			zz_order[i * 2] = tat[i].first;
			zz_order[i * 2 + 1] = tat[i].second;
		}

		//cl::Buffer test_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, deviceDataSize, (T *)m_test_img.data);
		//cl::Buffer reference_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, deviceDataSize, (T *)m_ref_img.data);
		cl::Buffer diff_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, deviceDataSize * sizeof(T), (T *)m_diff.data);
		cl::Buffer dictionary_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, dictionary.cols * dictionary.rows * sizeof(T), (T *)dictionary.data);
		cl::Buffer zigzag_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, zz_size, zz_order);
		cl::Buffer out_Buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, globalCycles * sizeof(T));

		try {
			std::ostringstream tmpStringStream;
			tmpStringStream << "-D IMAGE_W=" << nCols << " " \
				<< "-D IMAGE_H=" << nRows << " " \
				<< "-D STRIDE=" << stride << " " \
				<< "-D PATCHSIZE=" << m_params.getPatchSize() << " " \
				<< "-D LUM_INVARIANT_PATCH=" << m_params.lumInvariantPatch() << " " \
				<< "-D CONSTRAST_INVARIANT_PATCH=" << m_params.contrastInvariantPatch() << " " \
				<< "-D DICTIONARY_SIZE=" << m_params.getDictionarySize() << " " \
				<< "-D NCOLS=" << nCols << " " \
				<< "-D NROWS=" << nRows << " " \
				<< "-D NEIGHBOUR_RADIUS=" << neighborRadius;

			std::string compilerOptions = tmpStringStream.str();

			m_program.build(m_devicesVector, compilerOptions.c_str());

			cl::Kernel kernel = cl::Kernel(m_program, "bow");

			// Set the kernel arguments

			kernel.setArg<cl::Buffer>(0, out_Buffer);
			kernel.setArg<cl::Buffer>(1, diff_Buffer);
			kernel.setArg<cl::Buffer>(2, dictionary_Buffer);
			kernel.setArg<cl::Buffer>(3, zigzag_Buffer);



			//cl::NDRange localSize = cl::NDRange(wgWidth, wgHeight);
			cl::NDRange globalSize = cl::NDRange(nCols, nRows);

			// Execute the kernel
#ifdef PROFILE
			cl::Event prof_event;

			m_queue.enqueueNDRangeKernel(kernel_ssim, cl::NullRange, globalSize, localSize, \
				NULL, &prof_event);

			prof_event.setCallback(CL_COMPLETE, &CLInfo::kernelRunTimeCallBack, NULL);
			//m_queue.finish();
			prof_event.wait();
			//cl_ulong start = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			//cl_ulong end = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
			//std::cout << "Kernel runtime " <<  1.e-6 * (end-start) << " ms\n";
#else
			m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
			m_queue.finish();
#endif

			cv::Mat outMat(nRows, nCols, cv_mat_type);
			m_queue.enqueueReadBuffer(out_Buffer, CL_TRUE, 0, globalCycles * sizeof(T), artifactClusterIndices.data);

		}
		catch (cl::Error error) {
			if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devicesVector[0]) << std::endl;
			}
			std::cerr << "ERROR: private_compute()\n";
			throw;
		}
		free(zz_order);

		int nClusters = m_params.getDictionarySize();

		//cv::Mat H(nRows, nCols, CV_64FC(nClusters));

		H_vec = std::vector<cv::Mat>(32);//reserve(nClusters)

		for (std::vector<cv::Mat>::iterator it = H_vec.begin(); it != H_vec.end(); ++it)
			(*it) = cv::Mat::zeros(nRows, nCols, cv_mat_type);

		cv::Mat gaussianKernel = cv::Mat(1, neighborRadius * 2 + 1, cv_mat_type);
		T *gauss_Ptr = (T *)gaussianKernel.data;

		for (int i = -neighborRadius; i <= neighborRadius; i++) {
			*gauss_Ptr++ = std::exp((-(i*i)) / std::pow(0.5 * neighborRadius, 2.0));
		}

		gaussianKernel = gaussianKernel.t() * gaussianKernel;

		cv::Mat expandedImg = cv::Mat::zeros(2 * neighborRadius + nRows, 2 * neighborRadius + nCols, cv_mat_type);
		artifactClusterIndices.copyTo(expandedImg(cv::Rect(neighborRadius, neighborRadius, nCols, nRows)));
		

		cv::Mat H_data = cv::Mat::zeros(32 * nRows, nCols, cv_mat_type);

		size_t bufferSize = (2 * neighborRadius + nRows) * (2 * neighborRadius + nCols);
		size_t H_vec_size = 32 * nRows * nCols;
		size_t gaussianBufferSize = (neighborRadius * 2 + 1) * (neighborRadius * 2 + 1);
		size_t outputBufferSize;

		cl::Buffer expanded_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bufferSize * sizeof(T), (T *)expandedImg.data);
		cl::Buffer H_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, H_vec_size * sizeof(T), (T *)H_data.data);
		cl::Buffer gaussian_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, gaussianBufferSize * sizeof(T), (T *)gaussianKernel.data);
		cl::Buffer out_Buffer2 = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, H_vec_size * sizeof(T));

		try {
			cl::Kernel kernel = cl::Kernel(m_program, "bow2");

			// Set the kernel arguments
			kernel.setArg<cl::Buffer>(0, expanded_Buffer);
			kernel.setArg<cl::Buffer>(1, H_Buffer);
			kernel.setArg<cl::Buffer>(2, gaussian_Buffer);
			kernel.setArg<cl::Buffer>(3, out_Buffer2);

			cl::NDRange globaltime = cl::NDRange(nCols, nRows);

			// Execute the kernel
#ifdef PROFILE
			cl::Event prof_event;

			m_queue.enqueueNDRangeKernel(kernel_ssim, cl::NullRange, globalSize, localSize, \
				NULL, &prof_event);

			prof_event.setCallback(CL_COMPLETE, &CLInfo::kernelRunTimeCallBack, NULL);
			//m_queue.finish();
			prof_event.wait();
			//cl_ulong start = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			//cl_ulong end = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
			//std::cout << "Kernel runtime " <<  1.e-6 * (end-start) << " ms\n";
#else
			m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, globaltime);
			m_queue.finish();
#endif

			cv::Mat outMat(32 * nRows, nCols, cv_mat_type);
			m_queue.enqueueReadBuffer(out_Buffer2, CL_TRUE, 0, H_vec_size * sizeof(T), outMat.data);

			for (int i = 0; i < H_vec.size(); i++) {
				cv::Mat tmp = outMat(cv::Rect(0, nRows * i, nCols, nRows));
				tmp.copyTo(H_vec[i]);
			}

		
		}
		catch (cl::Error error) {
			if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devicesVector[0]) << std::endl;
			}
			std::cerr << "ERROR: private_compute()\n";
			throw;
		}


		for (unsigned int i = 0; i < H_vec.size(); i++)
			cv::resize(H_vec[i], H_vec[i], m_diff.size(), 0.0, 0.0, CV_INTER_LINEAR);


	/*	for (int i = 0; i < 32; i++) {
			cv::imshow("test", H_vec[i]);
			cv::waitKey();
		}*/
		//GPLib::writeCvMatToFile<double>(H_vec[0], "gaussianKernel.yml", true);
	}


	
	//*/
}
