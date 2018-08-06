#include <fstream>
#include <algorithm>
#include <opencv2/contrib/contrib.hpp>
#include "ssim_lum.hpp"

namespace dip {
    template void SSIM_LUM::private_compute<double>(int cv_mat_type) throw (std::runtime_error, cl::Error);
    template void SSIM_LUM::private_compute<float>(int cv_mat_type) throw (std::runtime_error, cl::Error);


    SSIM_LUM::SSIM_LUM(cv::Mat &testImg, cv::Mat &referenceImg, int platformNum, \
                    cl_device_type deviceType, int deviceNum) throw (std::runtime_error) {

        m_platformNum = platformNum;
        m_deviceNum = deviceNum;
        m_deviceType = deviceType;
        m_image_width = m_image_height = 0;

        m_sigma = 1.5;
        m_k_1 = 0.01;
        m_k_2 = 0.03;

        m_test_Img = testImg;
        m_reference_Img = referenceImg;

        if (m_test_Img.rows == 0 || m_test_Img.cols == 0)
            throw std::runtime_error("ERROR: setSources() - (m_test_Img.rows == 0 || m_test_Img.cols == 0)");
        if (m_test_Img.rows != m_reference_Img.rows || m_test_Img.cols != m_reference_Img.cols)
            throw std::runtime_error("ERROR: setSources() - (m_test_Img.rows != m_reference_Img.rows || m_test_Img.cols != m_reference_Img.cols)");
    }

    void SSIM_LUM::compute() throw (std::runtime_error) {
        m_image_width = m_test_Img.cols;
        m_image_height = m_test_Img.rows;

        m_filterSize = 2 * ((int) ceil(m_sigma * 3.0)) + 1;

        //if (m_filterSize > 15)
            fftw_convolution();
        //else {
            /*try {
                initCL();

                if (m_dev_doubleSupport)
                    private_compute<double>(CV_64FC1);
                else
                   private_compute<float>(CV_32FC1);
            } catch(...) {
               throw;
            }//*/
        //}
    }

    void SSIM_LUM::initCL() throw (std::runtime_error, cl::Error){
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

            cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) m_platform(), 0};
            m_context = cl::Context(m_deviceType, cprops);

            std::vector<cl::Device> tmp_devicesList = m_context.getInfo<CL_CONTEXT_DEVICES>();

            try {
                m_device = tmp_devicesList.at(m_deviceNum);
                m_dev_doubleSupport = dip::CLInfo::hasDoublePrecision(m_device);

                m_clSrcFileName = "ssim_lum.cl";

#ifdef _MSC_VER
                std::ifstream sourceFile("src\\" + m_clSrcFileName);
#else
                std::ifstream sourceFile("cl/" + m_clSrcFileName);
#endif

                std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), \
                                       (std::istreambuf_iterator<char>()));

                cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
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
        } catch (...) {
            throw;
        }
    }


    template <typename T>
    void SSIM_LUM::private_compute(int cv_mat_type) throw (std::runtime_error, cl::Error) {

		m_reference_Img = GPLib::get_luminance(m_reference_Img);
		m_test_Img = GPLib::get_luminance(m_test_Img);

        // create 2d gaus
        cv::Mat gaussianKernel = cv::getGaussianKernel(m_filterSize, m_sigma, cv_mat_type);
        gaussianKernel = gaussianKernel * gaussianKernel.t();

        m_test_Img.convertTo(m_test_Img, cv_mat_type);
        m_reference_Img.convertTo(m_reference_Img, cv_mat_type);

        m_test_Img *= (1.0/255.0);
        m_reference_Img *= (1.0/255.0);

        // Work group size
        size_t wgWidth = 16;
        size_t wgHeight = 16;

        int M = m_image_height + (gaussianKernel.rows - 1)*2;
        int N = m_image_width + (gaussianKernel.cols - 1)*2;

        M = roundUp(M, wgHeight);
        N = roundUp(N, wgWidth);

        int padd_M = (int) std::ceil((M - m_image_height) / 2.0 );
        int padd_N = (int) std::ceil((N - m_image_width) / 2.0 );

//         std::cout << "M: " << M << "   N: " << N << std::endl;
//         std::cout << "padd_M: " << padd_M << "   padd_N: " << padd_N << std::endl;

        cv::copyMakeBorder(m_test_Img, m_test_Img, padd_M, padd_M, padd_N, padd_N, cv::BORDER_REPLICATE);
        cv::copyMakeBorder(m_reference_Img, m_reference_Img, padd_M, padd_M, padd_N, padd_N, cv::BORDER_REPLICATE);

//         std::cout << "m_test_Img.size() = " << m_test_Img.size() << std::endl;
//         std::cout << "m_reference_Img.size() = " << m_reference_Img.size() << std::endl;

        // Device buffer memory size
        size_t deviceDataSize = M * N * sizeof(T);

        cl::Buffer test_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, deviceDataSize, (T *) m_test_Img.data);
        cl::Buffer reference_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, deviceDataSize, (T *) m_reference_Img.data);
        cl::Buffer filter_Buffer = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, m_filterSize*m_filterSize*sizeof(T), (T *) gaussianKernel.data);
        cl::Buffer out_Buffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, deviceDataSize);

        try {
            T L = 1.0;
            T C1 = static_cast<T>((std::pow(m_k_1 * L, 2.0)));

            std::ostringstream tmpStringStream;
            tmpStringStream << "-D IMAGE_W=" << N << " " \
                            << "-D IMAGE_H=" << M << " " \
                            << "-D FILTER_SIZE=" << m_filterSize << " " \
                            << "-D HALF_FILTER_SIZE=" << m_filterSize/2 << " " \
                            << "-D TWICE_HALF_FILTER_SIZE=" << (m_filterSize/2) * 2 << " " \
                            << "-D HALF_FILTER_SIZE_IMAGE_W=" << (m_filterSize/2) * N << " " \
                            << "-D C1=" << C1;

            std::string compilerOptions = tmpStringStream.str();

//             std::cout << compilerOptions << std::endl;

            m_program.build(m_devicesVector, compilerOptions.c_str());

            size_t localMemSize = ( wgWidth + 2 * (m_filterSize / 2) ) * ( wgHeight + 2 * (m_filterSize / 2) );
            cl::LocalSpaceArg localMem_test = cl::Local(sizeof(T) * localMemSize);
            cl::LocalSpaceArg localMem_reference = cl::Local(sizeof(T) * localMemSize);

            cl::Kernel kernel_ssim =  cl::Kernel(m_program, "ssim_lum");

            // Set the kernel arguments
            kernel_ssim.setArg<cl::Buffer>(0, test_Buffer);
            kernel_ssim.setArg<cl::Buffer>(1, reference_Buffer);
            kernel_ssim.setArg<cl::Buffer>(2, out_Buffer);
            kernel_ssim.setArg<cl::Buffer>(3, filter_Buffer);
            kernel_ssim.setArg<cl::LocalSpaceArg>(4, localMem_test);
            kernel_ssim.setArg<cl::LocalSpaceArg>(5, localMem_reference);


            cl::NDRange localSize = cl::NDRange(wgWidth, wgHeight);
            cl::NDRange globalSize = cl::NDRange(N, M);

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
            m_queue.enqueueNDRangeKernel(kernel_ssim, cl::NullRange, globalSize, localSize, \
                                        NULL, NULL);
            m_queue.finish();
#endif

            cv::Mat outMat(M, N, cv_mat_type);
            m_queue.enqueueReadBuffer(out_Buffer, CL_TRUE, 0, deviceDataSize, outMat.data);

            cv::Rect myROI(padd_N, padd_M, m_image_width, m_image_height);

            m_result = outMat(myROI);
        } catch (cl::Error error) {
            if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devicesVector[0]) << std::endl;
            }
            std::cerr << "ERROR: private_compute()\n";
            throw;
        }
    }


    double SSIM_LUM::fftw_convolution() {

		m_reference_Img = GPLib::get_luminance(m_reference_Img);
		m_test_Img = GPLib::get_luminance(m_test_Img);

        m_test_Img.convertTo(m_test_Img, CV_64FC1);
        m_reference_Img.convertTo(m_reference_Img, CV_64FC1);

        cv::Mat gaussianKernel = cv::getGaussianKernel(m_filterSize, m_sigma, CV_64FC1);
        gaussianKernel = gaussianKernel * gaussianKernel.t();
		
		cv::Mat mu1mat = cv::Mat::zeros(m_image_height, m_image_width, CV_64FC1);
		cv::Mat mu2mat = cv::Mat::zeros(m_image_height, m_image_width, CV_64FC1);
		cv::filter2D(m_test_Img, mu1mat, -1, gaussianKernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
		cv::filter2D(m_reference_Img, mu2mat, -1, gaussianKernel, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

		//GPLib::writeCvMatToFile<double>(mu2mat, "matrixes/mu2mat.yml", true);
		cv::Mat mu1_sqr = mu1mat.mul(mu1mat);
		cv::Mat mu2_sqr = mu2mat.mul(mu2mat);
		cv::Mat mu21 = mu2mat.mul(mu1mat);


		double C1 = std::pow(m_k_1*1.0, 2.0);
		m_result = (2.0 * mu21 + C1) / (mu1_sqr + mu2_sqr + C1);
		return 0.0;



        int M = m_image_height + gaussianKernel.rows - 1;
        int N = m_image_width + gaussianKernel.cols - 1;

        if ((M & (M - 1)) == 0 )
            M = nextPowerOf2(M);
        else if ((N & (N - 1)) == 0 )
            N = nextPowerOf2(N);

        cv::copyMakeBorder(m_test_Img, m_test_Img, 0, M - m_image_height, 0, N - m_image_width, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(m_reference_Img, m_reference_Img, 0, M - m_image_height, 0, N - m_image_width, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(gaussianKernel, gaussianKernel, 0, M - gaussianKernel.rows, 0, N - gaussianKernel.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        size_t numOfPixels = M * N;

        // allocate input arrays
        fftw_complex *in_testImage     = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *in_refImage      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *in_Kernel        = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *in_test_Mul_test = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *in_ref_Mul_ref   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *in_test_Mul_ref  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);


        // assign values to real parts (values between 0 and MaxRGB)
        for(unsigned int i = 0; i < numOfPixels; i++) {
            // save as real numbers
            in_testImage[i][0]  = *((double *) m_test_Img.data + i) / 255.0;
            in_Kernel[i][0] = *((double *) gaussianKernel.data + i);

            // save as real numbers
            in_testImage[i][0]     = *((double *) m_test_Img.data + i) / 255.0;
            in_refImage[i][0]      = *((double *) m_reference_Img.data + i) / 255.0;
            in_Kernel[i][0]        = *((double *) gaussianKernel.data + i);

            in_test_Mul_test[i][0] = in_testImage[i][0] * in_testImage[i][0];
            in_ref_Mul_ref[i][0]   = in_refImage[i][0] *in_refImage[i][0];
            in_test_Mul_ref[i][0]  = in_testImage[i][0] * in_refImage[i][0];
        }

        // allocate output arrays
        fftw_complex *out_testImage     = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *out_refImage      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *out_Kernel        = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *out_test_Mul_test = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *out_ref_Mul_ref   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);
        fftw_complex *out_test_Mul_ref  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);

        // create FTT plans
        fftw_plan ftt_testImage     = fftw_plan_dft_2d(M, N, in_testImage, out_testImage, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan ftt_refImage      = fftw_plan_dft_2d(M, N, in_refImage, out_refImage, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan ftt_Kernel        = fftw_plan_dft_2d(M, N, in_Kernel, out_Kernel, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan ftt_test_Mul_test = fftw_plan_dft_2d(M, N, in_test_Mul_test, out_test_Mul_test, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan ftt_ref_Mul_ref   = fftw_plan_dft_2d(M, N, in_ref_Mul_ref, out_ref_Mul_ref, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan ftt_test_Mul_ref  = fftw_plan_dft_2d(M, N, in_test_Mul_ref, out_test_Mul_ref, FFTW_FORWARD, FFTW_ESTIMATE);

        // perform FORWARD fft
        fftw_execute(ftt_testImage);
        fftw_execute(ftt_refImage);
        fftw_execute(ftt_Kernel);
        fftw_execute(ftt_test_Mul_test);
        fftw_execute(ftt_ref_Mul_ref);
        fftw_execute(ftt_test_Mul_ref);

        fftw_destroy_plan(ftt_testImage);
        fftw_destroy_plan(ftt_refImage);
        fftw_destroy_plan(ftt_Kernel);
        fftw_destroy_plan(ftt_test_Mul_test);
        fftw_destroy_plan(ftt_ref_Mul_ref);
        fftw_destroy_plan(ftt_test_Mul_ref);

        // transform imaginary number to phase and magnitude and save to output
        for(unsigned int i = 0; i < numOfPixels; i++) {
            // real parts
            double real_test_kernel      = (out_testImage[i][0] * out_Kernel[i][0]) - (out_testImage[i][1] * out_Kernel[i][1]);
            double real_ref_kernel       = (out_refImage[i][0] * out_Kernel[i][0]) - (out_refImage[i][1] * out_Kernel[i][1]);
            double real_testMtest_kernel = (out_test_Mul_test[i][0] * out_Kernel[i][0]) - (out_test_Mul_test[i][1] * out_Kernel[i][1]);
            double real_refMref_kernel   = (out_ref_Mul_ref[i][0] * out_Kernel[i][0]) - (out_ref_Mul_ref[i][1] * out_Kernel[i][1]);
            double real_testMref_kernel  = (out_test_Mul_ref[i][0] * out_Kernel[i][0]) - (out_test_Mul_ref[i][1] * out_Kernel[i][1]);

            // imaginary parts
            double imag_test_kernel      = (out_testImage[i][0] * out_Kernel[i][1]) + (out_testImage[i][1] * out_Kernel[i][0]);
            double imag_ref_kernel       = (out_refImage[i][0] * out_Kernel[i][1]) + (out_refImage[i][1] * out_Kernel[i][0]);
            double imag_testMtest_kernel = (out_test_Mul_test[i][0] * out_Kernel[i][1]) + (out_test_Mul_test[i][1] * out_Kernel[i][0]);
            double imag_refMref_kernel   = (out_ref_Mul_ref[i][0] * out_Kernel[i][1]) + (out_ref_Mul_ref[i][1] * out_Kernel[i][0]);
            double imag_testMref_kernel  = (out_test_Mul_ref[i][0] * out_Kernel[i][1]) + (out_test_Mul_ref[i][1] * out_Kernel[i][0]);

            // magnitude
            double mag_test      = sqrt((real_test_kernel * real_test_kernel) + (imag_test_kernel * imag_test_kernel));
            double mag_ref       = sqrt((real_ref_kernel * real_ref_kernel) + (imag_ref_kernel * imag_ref_kernel));
            double mag_testMtest = sqrt((real_testMtest_kernel * real_testMtest_kernel) + (imag_testMtest_kernel * imag_testMtest_kernel));
            double mag_refMref   = sqrt((real_refMref_kernel * real_refMref_kernel) + (imag_refMref_kernel * imag_refMref_kernel));
            double mag_testMref  = sqrt((real_testMref_kernel * real_testMref_kernel) + (imag_testMref_kernel * imag_testMref_kernel));

            // phase
            double phase_test      = std::atan2(imag_test_kernel, real_test_kernel);
            double phase_ref       = std::atan2(imag_ref_kernel, real_ref_kernel);
            double phase_testMtest = std::atan2(imag_testMtest_kernel, real_testMtest_kernel);
            double phase_refMref   = std::atan2(imag_refMref_kernel, real_refMref_kernel);
            double phase_testMref  = std::atan2(imag_testMref_kernel, real_testMref_kernel);

            in_testImage[i][0] = (mag_test * cos(phase_test));
            in_testImage[i][1] = (mag_test * sin(phase_test));

            in_refImage[i][0] = (mag_ref * cos(phase_ref));
            in_refImage[i][1] = (mag_ref * sin(phase_ref));

            in_test_Mul_test[i][0] = (mag_testMtest * cos(phase_testMtest));
            in_test_Mul_test[i][1] = (mag_testMtest * sin(phase_testMtest));

            in_ref_Mul_ref[i][0] = (mag_refMref * cos(phase_refMref));
            in_ref_Mul_ref[i][1] = (mag_refMref * sin(phase_refMref));

            in_test_Mul_ref[i][0] = (mag_testMref * cos(phase_testMref));
            in_test_Mul_ref[i][1] = (mag_testMref * sin(phase_testMref));
        }

        fftw_free(in_Kernel);
        fftw_free(out_Kernel);

        // create IFTT plans
        fftw_plan ifft_testImage     = fftw_plan_dft_2d(M, N, in_testImage, out_testImage, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_plan ifft_refImage      = fftw_plan_dft_2d(M, N, in_refImage, out_refImage, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_plan iftt_test_Mul_test = fftw_plan_dft_2d(M, N, in_test_Mul_test, out_test_Mul_test, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_plan iftt_ref_Mul_ref   = fftw_plan_dft_2d(M, N, in_ref_Mul_ref, out_ref_Mul_ref, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_plan iftt_test_Mul_ref  = fftw_plan_dft_2d(M, N, in_test_Mul_ref, out_test_Mul_ref, FFTW_BACKWARD, FFTW_ESTIMATE);

        // perform ifft
        fftw_execute(ifft_testImage);
        fftw_execute(ifft_refImage);
        fftw_execute(iftt_test_Mul_test);
        fftw_execute(iftt_ref_Mul_ref);
        fftw_execute(iftt_test_Mul_ref);

        fftw_destroy_plan(ifft_testImage);
        fftw_destroy_plan(ifft_refImage);
        fftw_destroy_plan(iftt_test_Mul_test);
        fftw_destroy_plan(iftt_ref_Mul_ref);
        fftw_destroy_plan(iftt_test_Mul_ref);

        fftw_free(in_testImage);
        fftw_free(in_refImage);
        fftw_free(in_test_Mul_test);
        fftw_free(in_ref_Mul_ref);
        fftw_free(in_test_Mul_ref);

        // save real parts to output
        m_result = cv::Mat::zeros(M, N, CV_64FC1);
        double scale = 1.0 / numOfPixels;
        for(unsigned int i = 0; i < numOfPixels; i++) {
            double L = 1.0;
            double C1 = std::pow(m_k_1*L, 2.0);

            double mu1     = out_testImage[i][0] * scale;
            double mu2     = out_refImage[i][0] * scale;
            double mu1_sq  = mu1 * mu1;
            double mu2_sq  = mu2 * mu2;
            double mu1_mu2 = mu1 * mu2;

            double lum   = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1);

            *((double *) m_result.data + i) = lum;
        }

        fftw_free(out_testImage);
        fftw_free(out_refImage);
        fftw_free(out_test_Mul_test);
        fftw_free(out_ref_Mul_ref);
        fftw_free(out_test_Mul_ref);

        int padC_m = (int) ceil((m_filterSize - 1) / 2);

        m_result.adjustROI(-padC_m,-padC_m, -padC_m, -padC_m);

        return cv::mean(m_result)[0];
    }


    unsigned int SSIM_LUM::roundUp(unsigned int value, unsigned int multiple) const {
        // Determine how far past the nearest multiple the value is
        unsigned int remainder = value % multiple;

        // Add the difference to make the value a multiple
        if(remainder != 0)
            value += (multiple-remainder);
        return value;
    }


    int SSIM_LUM::nextPowerOf2(int32_t number) const {
        number--;
        number |= number >> 1;
        number |= number >> 2;
        number |= number >> 4;
        number |= number >> 8;
        number |= number >> 16;
        number++;

        return number;
    }


    cv::Mat SSIM_LUM::normalizeMat(const cv::Mat& inputMat) {
        double minVal, maxVal;
        cv::minMaxLoc(inputMat, &minVal, &maxVal); //find minimum and maximum intensities
        cv::Mat out = cv::Mat(inputMat.rows, inputMat.cols, CV_8UC1);
        inputMat.convertTo(out, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

        return out;
    }
}
