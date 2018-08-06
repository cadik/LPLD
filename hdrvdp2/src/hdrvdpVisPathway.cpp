#include <functional>
#include <cmath>
#include "hdrvdpVisPathway.hpp"

using namespace std;

namespace dip {
    HDRVDP_VisPathway::HDRVDP_VisPathway(const cv::Mat &image, const MetricParams &metric_par, clClass *clObj) {
        m_clObj = clObj;

        m_imgWidth = image.cols;
        m_imgHeight = image.rows;
        m_imgChannels = image.channels();

        initLambda();

        m_rho2 = CreateCycdegImage(m_imgWidth, m_imgHeight, metric_par.pix_per_degree);

        CreateMtf(metric_par);
        load_LMSR_S();
        photorecNonLinear(metric_par);
        opticalTransferFun(image, metric_par);
        photoSpectralSensitivity(metric_par);
    }


    cv::Mat HDRVDP_VisPathway::CreateCycdegImage(int imgWidth, int imgHeight, double pix_per_deg) { //const MetricParams *metric_par) {
        cv::Mat xx, yy, result;
        double nyquist_freq = 0.5 * pix_per_deg;
        double freq_stepX = nyquist_freq / imgWidth;
        double freq_stepY = nyquist_freq / imgHeight;

        cv::hconcat(HDRVDP_helper::createSpace(0, nyquist_freq-freq_stepX, imgWidth, LINEAR), \
            HDRVDP_helper::createSpace(-nyquist_freq, -freq_stepX, imgWidth, LINEAR), xx);

        cv::hconcat(HDRVDP_helper::createSpace(0, nyquist_freq-freq_stepY, imgHeight, LINEAR), \
            HDRVDP_helper::createSpace(-nyquist_freq, -freq_stepY, imgHeight, LINEAR), yy);

        cv::Mat X, Y;

        cv::repeat(xx.reshape(1,1), yy.total(), 1, X);
        cv::repeat(yy.reshape(1,1).t(), 1, xx.total(), Y);

        cv::pow(X, 2, X);
        cv::pow(Y, 2, Y);

        cv::sqrt(X+Y, result);
        result.convertTo(result, CV_64FC1);

        return result;
    }


    void HDRVDP_VisPathway::CreateMtf(const MetricParams &metric_par) {
        if (metric_par.do_mtf) {

            if (m_clObj->devSupportDouble()) {

                m_mtf_filter = cv::Mat::zeros(m_rho2.rows, m_rho2.cols, CV_64FC1);

                int deviceDataSize = m_mtf_filter.rows * m_mtf_filter.cols * m_mtf_filter.channels() * sizeof(double);

                cl::Buffer inBuffer1(*m_clObj->getContext(), CL_MEM_READ_ONLY, deviceDataSize);
                cl::Buffer inBuffer2(*m_clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
                cl::Buffer inBuffer3(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(double));
                cl::Buffer inBuffer4(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(double));

                cl::Kernel kernel(*m_clObj->getProgram(), "hdrvdp_mtfCL");

                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer1, CL_TRUE, 0, deviceDataSize, m_rho2.data);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_mtf_filter.data);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer3, CL_TRUE, 0, 4 * sizeof(double), metric_par.mtf_params_a);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer4, CL_TRUE, 0, 4 * sizeof(double), metric_par.mtf_params_b);

                kernel.setArg(0, inBuffer1);
                kernel.setArg(1, inBuffer2);
                kernel.setArg(2, inBuffer3);
                kernel.setArg(3, inBuffer4);

                (*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, \
                    cl::NDRange(m_mtf_filter.rows * m_mtf_filter.cols * m_mtf_filter.channels()));

                (*m_clObj->getQueue()).finish();

                (*m_clObj->getQueue()).enqueueReadBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_mtf_filter.data);
            }
            else {
                m_mtf_filter = cv::Mat::zeros(m_rho2.rows, m_rho2.cols, CV_32FC1);

                int deviceDataSize = m_mtf_filter.rows * m_mtf_filter.cols * m_mtf_filter.channels() * sizeof(float);

                cl::Buffer inBuffer1(*m_clObj->getContext(), CL_MEM_READ_ONLY, deviceDataSize);
                cl::Buffer inBuffer2(*m_clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
                cl::Buffer inBuffer3(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(float));
                cl::Buffer inBuffer4(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(float));

                cl::Kernel kernel(*m_clObj->getProgram(), "hdrvdp_mtfCL");

                m_rho2.convertTo(m_rho2, CV_32FC1);

                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer1, CL_TRUE, 0, deviceDataSize, m_rho2.data);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_mtf_filter.data);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer3, CL_TRUE, 0, 4 * sizeof(float), metric_par.mtf_params_a_f);
                (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer4, CL_TRUE, 0, 4 * sizeof(float), metric_par.mtf_params_b_f);

                kernel.setArg(0, inBuffer1);
                kernel.setArg(1, inBuffer2);
                kernel.setArg(2, inBuffer3);
                kernel.setArg(3, inBuffer4);

                (*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, \
                    cl::NDRange(m_mtf_filter.rows * m_mtf_filter.cols * m_mtf_filter.channels()));

                (*m_clObj->getQueue()).finish();

                (*m_clObj->getQueue()).enqueueReadBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_mtf_filter.data);

                m_rho2.convertTo(m_rho2, CV_64FC1);
            }

        } else
            m_mtf_filter = cv::Mat::ones(m_rho2.size(), CV_64FC1);

        m_mtf_filter.convertTo(m_mtf_filter, CV_64FC1);
    }


    void HDRVDP_VisPathway::load_LMSR_S() throw (std::runtime_error) {
        unsigned int rows;

        std::ifstream file("LMSR_S.csv");
        rows = static_cast<unsigned int>(std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n'));
        
        if (rows == 0) {
            throw std::runtime_error("ERROR: LMSR_S.csv not found!");
        }
        
        file.clear();
        file.seekg(0, std::ios::beg);

        LMSR_S = cv::Mat(rows, 4, CV_64FC1);

        double *dataPtr = (double*) LMSR_S.data;

        int i = -1;
        for(dip::CSVIterator loop(file);loop != dip::CSVIterator();++loop) {
            dataPtr[++i] = atof((*loop)[0].c_str());
            dataPtr[++i] = atof((*loop)[1].c_str());
            dataPtr[++i] = atof((*loop)[2].c_str());
            dataPtr[++i] = atof((*loop)[3].c_str());
        }
    }

	std::vector<double> HDRVDP_VisPathway::linear_interpolation_GPU(std::map<double, double> &X_V, int cycles, double * ptrToData) {
		std::vector<double> Vq(cycles);

		typedef std::map<double, double>::const_iterator i_t;

		int Xv_size = cycles;
		int X_V_size = X_V.size();

		if (m_clObj->devSupportDouble()) {
			std::vector<double> XVvectorFirst;
			std::vector<double> XVvectorSecond;
			
			for (std::map<double, double>::iterator it = X_V.begin(); it != X_V.end(); ++it) {
				
				XVvectorFirst.push_back(it->first);
				XVvectorSecond.push_back(it->second);
				
			}
			std::vector<int> info;
			info.push_back(X_V_size);

			
			//tmp = &Xv[0];
			cl::Buffer Xv_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, Xv_size * sizeof(double));
			cl::Buffer X_First_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(double));
			cl::Buffer X_Second_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(double));
			cl::Buffer out_Buffer(*m_clObj->getContext(), CL_MEM_READ_WRITE, Xv_size * sizeof(double));
			cl::Buffer info_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, info.size() * sizeof(int));
			//cl::Kernel kernel(*m_clObj->getProgram(), "hdrvdp_mtfCL");
			cl::Kernel kernel(*m_clObj->getProgram(), "linear_interpolation_GPU");
			(*m_clObj->getQueue()).enqueueWriteBuffer(Xv_Buffer, CL_TRUE, 0, Xv_size * sizeof(double), ptrToData);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_First_Buffer, CL_TRUE, 0, X_V_size * sizeof(double), (double*)&XVvectorFirst[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_Second_Buffer, CL_TRUE, 0, X_V_size * sizeof(double), (double*)&XVvectorSecond[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(info_Buffer, CL_TRUE, 0, info.size() * sizeof(int), (int*)&info[0]);

			kernel.setArg(0, Xv_Buffer);
			kernel.setArg(1, X_First_Buffer);
			kernel.setArg(2, X_Second_Buffer);
			kernel.setArg(3, out_Buffer);
			kernel.setArg(4, info_Buffer);

			(*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Xv_size));

			(*m_clObj->getQueue()).finish();
			//double * BQ = (double*)calloc(Xv_size, sizeof(double));
			(*m_clObj->getQueue()).enqueueReadBuffer(out_Buffer, CL_TRUE, 0, Xv_size* sizeof(double), &Vq[0]);

		}
		else {
			std::vector<float> XVvectorFirst;
			std::vector<float> XVvectorSecond;
			

			for (std::map<double, double>::iterator it = X_V.begin(); it != X_V.end(); ++it) {
				XVvectorFirst.push_back(it->first);
				XVvectorSecond.push_back(it->second);
			}
			std::vector<int> info;
			info.push_back(X_V_size);

			
			

			//tmp = &Xv[0];
			cl::Buffer Xv_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, Xv_size * sizeof(float));
			cl::Buffer X_First_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(float));
			cl::Buffer X_Second_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(float));
			cl::Buffer out_Buffer(*m_clObj->getContext(), CL_MEM_READ_WRITE, Xv_size * sizeof(float));
			cl::Buffer info_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, info.size() * sizeof(int));
			
			cl::Kernel kernel(*m_clObj->getProgram(), "linear_interpolation_GPU");

			float * tmp = (float*)calloc(Xv_size, sizeof(float));
			std::copy(ptrToData, ptrToData + Xv_size, tmp);
			(*m_clObj->getQueue()).enqueueWriteBuffer(Xv_Buffer, CL_TRUE, 0, Xv_size * sizeof(float), tmp);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_First_Buffer, CL_TRUE, 0, X_V_size * sizeof(float), (float*)&XVvectorFirst[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_Second_Buffer, CL_TRUE, 0, X_V_size * sizeof(float), (float*)&XVvectorSecond[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(info_Buffer, CL_TRUE, 0, info.size() * sizeof(int), (int*)&info[0]);

			kernel.setArg(0, Xv_Buffer);
			kernel.setArg(1, X_First_Buffer);
			kernel.setArg(2, X_Second_Buffer);
			kernel.setArg(3, out_Buffer);
			kernel.setArg(4, info_Buffer);

			(*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Xv_size));

			(*m_clObj->getQueue()).finish();
			
			std::vector<float> VqF(cycles);
			(*m_clObj->getQueue()).enqueueReadBuffer(out_Buffer, CL_TRUE, 0, Xv_size * sizeof(float), &VqF[0]);
			Vq = std::vector<double>(VqF.begin(), VqF.end());
		}

		return Vq;
	}

	std::vector<double> HDRVDP_VisPathway::getXvec_GPU(double* ptrToData,int cycles, int size) {

		std::vector<double> Xv_vec(0);

		/*if (*ptrToData < pow(10.0, pn_Y.at<double>(0)))
			*ptrToData = pow(10.0, pn_Y.at<double>(0));
		if (*ptrToData > pow(10.0, pn_Y.at<double>(pn_Y.rows * pn_Y.cols - 1)))
			*ptrToData = pow(10.0, pn_Y.at<double>(pn_Y.rows * pn_Y.cols - 1));

		*ptrToData = log10(*ptrToData);
		Xv_vec.push_back(*ptrToData);

		*ptrToData++;
		
		if (m_clObj->devSupportDouble()) {
			std::vector<double> XVvectorFirst;
			std::vector<double> XVvectorSecond;

			for (std::map<double, double>::iterator it = X_V.begin(); it != X_V.end(); ++it) {

				XVvectorFirst.push_back(it->first);
				XVvectorSecond.push_back(it->second);

			}
			std::vector<int> info;
			info.push_back(X_V_size);

			//tmp = &Xv[0];
			cl::Buffer Xv_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, Xv_size * sizeof(double));
			cl::Buffer X_First_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(double));
			cl::Buffer X_Second_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, X_V_size * sizeof(double));
			cl::Buffer out_Buffer(*m_clObj->getContext(), CL_MEM_READ_WRITE, Xv_size * sizeof(double));
			cl::Buffer info_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, info.size() * sizeof(int));
			//cl::Kernel kernel(*m_clObj->getProgram(), "hdrvdp_mtfCL");
			cl::Kernel kernel(*m_clObj->getProgram(), "linear_interpolation_GPU");
			(*m_clObj->getQueue()).enqueueWriteBuffer(Xv_Buffer, CL_TRUE, 0, Xv_size * sizeof(double), (double*)&Xv[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_First_Buffer, CL_TRUE, 0, X_V_size * sizeof(double), (double*)&XVvectorFirst[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(X_Second_Buffer, CL_TRUE, 0, X_V_size * sizeof(double), (double*)&XVvectorSecond[0]);
			(*m_clObj->getQueue()).enqueueWriteBuffer(info_Buffer, CL_TRUE, 0, info.size() * sizeof(int), (int*)&info[0]);

			kernel.setArg(0, Xv_Buffer);
			kernel.setArg(1, X_First_Buffer);
			kernel.setArg(2, X_Second_Buffer);
			kernel.setArg(3, out_Buffer);
			kernel.setArg(4, info_Buffer);

			(*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Xv_size));

			(*m_clObj->getQueue()).finish();
			//double * BQ = (double*)calloc(Xv_size, sizeof(double));
			(*m_clObj->getQueue()).enqueueReadBuffer(out_Buffer, CL_TRUE, 0, Xv_size * sizeof(double), &Vq[0]);

		}
		else {
		}*/
		return Xv_vec;
	}

    void HDRVDP_VisPathway::photorecNonLinear(const MetricParams &metric_par) {
        cv::Mat s_A, s_R, V, Xq, s_C;
        cl::Buffer csf_sa_Buffer, csf_sr_par_Buffer;
        cv::Mat cl = HDRVDP_helper::createSpace(-5.0, 5.0, 2048, dip::SPACE_TYPE::LOG);
        pn_Y = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
        int deviceDataSize;

        double* ptrCL = (double*) cl.data;
        double* ptrpn_Y = (double*) pn_Y.data;

        for( int i = 0; i < cl.rows * cl.cols; ++i)
            *ptrpn_Y++ = log10(*ptrCL++);

        if (m_clObj->devSupportDouble()) {
            deviceDataSize = cl.rows * cl.cols * sizeof(double);
            s_A = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
            s_R = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
            V = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
            Xq = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
            s_C = cv::Mat(cl.rows, cl.cols, CV_64FC1, 0.0);
        }
        else {
            cl.convertTo(cl, CV_32FC1);
            deviceDataSize = cl.rows * cl.cols * sizeof(float);
            s_A = cv::Mat(cl.rows, cl.cols, CV_32FC1, 0.0);
            s_R = cv::Mat(cl.rows, cl.cols, CV_32FC1, 0.0);
            V = cv::Mat(cl.rows, cl.cols, CV_32FC1, 0.0);
            Xq = cv::Mat(cl.rows, cl.cols, CV_32FC1, 0.0);
            s_C = cv::Mat(cl.rows, cl.cols, CV_32FC1, 0.0);
        }

        cl::Buffer la_Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, deviceDataSize);
        cl::Buffer s_A_Buffer(*m_clObj->getContext(), CL_MEM_WRITE_ONLY, deviceDataSize);
        cl::Buffer s_R_Buffer(*m_clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
        cl::Buffer V_Buffer(*m_clObj->getContext(), CL_MEM_WRITE_ONLY, deviceDataSize);
        cl::Buffer Xq_Buffer(*m_clObj->getContext(), CL_MEM_WRITE_ONLY, deviceDataSize);

        if (m_clObj->devSupportDouble()) {
            csf_sa_Buffer = cl::Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(double));
            csf_sr_par_Buffer = cl::Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, 6 * sizeof(double));
        }
        else {
            csf_sa_Buffer = cl::Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, 4 * sizeof(float));
            csf_sr_par_Buffer = cl::Buffer(*m_clObj->getContext(), CL_MEM_READ_ONLY, 6 * sizeof(float));
        }

        cl::Kernel kernel(*m_clObj->getProgram(), "jointRodConeSens_rodSens");

        (*m_clObj->getQueue()).enqueueWriteBuffer(la_Buffer, CL_TRUE, 0, deviceDataSize, cl.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(s_A_Buffer, CL_TRUE, 0, deviceDataSize, s_A.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(s_R_Buffer, CL_TRUE, 0, deviceDataSize, s_R.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(V_Buffer, CL_TRUE, 0, deviceDataSize, V.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(Xq_Buffer, CL_TRUE, 0, deviceDataSize, Xq.data);

        if (m_clObj->devSupportDouble()) {
            (*m_clObj->getQueue()).enqueueWriteBuffer(csf_sa_Buffer, CL_TRUE, 0, 4 * sizeof(double), metric_par.csf_sa);
            (*m_clObj->getQueue()).enqueueWriteBuffer(csf_sr_par_Buffer, CL_TRUE, 0, 6 * sizeof(double), metric_par.csf_sr_par);
        }
        else {
            (*m_clObj->getQueue()).enqueueWriteBuffer(csf_sa_Buffer, CL_TRUE, 0, 4 * sizeof(float), metric_par.csf_sa_f);
            (*m_clObj->getQueue()).enqueueWriteBuffer(csf_sr_par_Buffer, CL_TRUE, 0, 6 * sizeof(float), metric_par.csf_sr_par_f);
        }

        kernel.setArg(0, la_Buffer);
        kernel.setArg(1, s_A_Buffer);
        kernel.setArg(2, s_R_Buffer);
        kernel.setArg(3, V_Buffer);
        kernel.setArg(4, Xq_Buffer);
        kernel.setArg(5, csf_sa_Buffer);
        kernel.setArg(6, csf_sr_par_Buffer);
        if (m_clObj->devSupportDouble())
            kernel.setArg(7, metric_par.rod_sensitivity);
        else
            kernel.setArg(7, metric_par.rod_sensitivity_f);

        (*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cl.rows * cl.cols));

        (*m_clObj->getQueue()).finish();

        (*m_clObj->getQueue()).enqueueReadBuffer(s_A_Buffer, CL_TRUE, 0, deviceDataSize, s_A.data);
        (*m_clObj->getQueue()).enqueueReadBuffer(s_R_Buffer, CL_TRUE, 0, deviceDataSize, s_R.data);
        (*m_clObj->getQueue()).enqueueReadBuffer(V_Buffer, CL_TRUE, 0, deviceDataSize, V.data);
        (*m_clObj->getQueue()).enqueueReadBuffer(Xq_Buffer, CL_TRUE, 0, deviceDataSize, Xq.data);


        if (!m_clObj->devSupportDouble()) {
            cl.convertTo(cl, CV_64FC1);
            s_A.convertTo(s_A, CV_64FC1);
            s_R.convertTo(s_R, CV_64FC1);
            V.convertTo(V, CV_64FC1);
            Xq.convertTo(Xq, CV_64FC1);
            s_C.convertTo(s_C, CV_64FC1);
        }

        std::map<double, double> X_V;
        std::vector<double> Xv_vec;
        std::vector<double> Vq;
        for (int i = 0; i < cl.cols; i++) {
            X_V.insert(std::pair<double, double>(cl.at<double>(i), V.at<double>(i)));
            Xv_vec.push_back(Xq.at<double>(i));
        }

        Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec);
        std::transform(Vq.begin(), Vq.end(), Vq.begin(), std::bind1st(std::multiplies<double>(), 0.5));

        memcpy(s_C.data,Vq.data(),Vq.size()*sizeof(double));

        X_V.clear(); std::map<double, double>().swap(X_V);
        Xv_vec.clear(); std::vector<double>().swap(Xv_vec);
        Vq.clear(); std::vector<double>().swap(Vq);

        cl.release();

        pn_jnd[0] = build_jndspace_from_S(pn_Y, s_C) * metric_par.sensitivity_correction;
        pn_jnd[1] = build_jndspace_from_S(pn_Y, s_R) * metric_par.sensitivity_correction;
    }

    cv::Mat HDRVDP_VisPathway::hdrvdp_rod_sens(const cv::Mat &la, const MetricParams &metric_par) {
        cv::Mat out(la.rows, la.cols, CV_64FC1, 0.0);
        int deviceDataSize = la.rows * la.cols * sizeof(double);

        cl::Buffer inBuffer1(*m_clObj->getContext(), CL_MEM_READ_ONLY, deviceDataSize);
        cl::Buffer outBuffer(*m_clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
        cl::Buffer inBuffer2(*m_clObj->getContext(), CL_MEM_READ_ONLY, 6*sizeof(double));

        cl::Kernel kernel(*m_clObj->getProgram(), "rodSens");

        (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer1, CL_TRUE, 0, deviceDataSize, la.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(outBuffer, CL_TRUE, 0, deviceDataSize, out.data);
        (*m_clObj->getQueue()).enqueueWriteBuffer(inBuffer2, CL_TRUE, 0, 6*sizeof(double), metric_par.csf_sr_par);

        kernel.setArg(0, inBuffer1);
        kernel.setArg(1, outBuffer);
        kernel.setArg(2, inBuffer2);
        kernel.setArg(3, metric_par.rod_sensitivity);

        (*m_clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(la.rows * la.cols));

        (*m_clObj->getQueue()).finish();

        (*m_clObj->getQueue()).enqueueReadBuffer(outBuffer, CL_TRUE, 0, deviceDataSize, out.data);

        return out;
    }

    cv::Mat HDRVDP_VisPathway::build_jndspace_from_S(const cv::Mat &l, const cv::Mat &S) {
        cv::Mat dL(l.rows, l.cols, CV_64FC1, 0.0);
        cv::Mat out(l.rows, l.cols, CV_64FC1, 0.0);

        double* ptrl = (double*) l.data;
        double* ptrS = (double*) S.data;
        double* ptrdL = (double*) dL.data;
        double* ptrOut = (double*) out.data;

        double thr;

        for( int i = 0; i < l.rows * l.cols; ++i) {
            thr = pow(10.0, *ptrl) / *ptrS++;
            *ptrdL++ = 1 / thr * pow(10.0, *ptrl++) * log(10.0);
        }

        ptrl = (double*) l.data;
        ptrdL = (double*) dL.data;

        ptrOut++;
        for( int i = 0; i < (l.rows * l.cols) - 1; ++i) {
             *ptrOut = *(ptrOut - 1) + ((*ptrdL + *(ptrdL + 1)) * (*(ptrl + 1)- *ptrl)) / 2.0;
             
             ptrOut++;
             ptrl++;
             ptrdL++;
        }


        return out;
    }

    void HDRVDP_VisPathway::opticalTransferFun(const cv::Mat &img, const MetricParams &metric_par) {
        double* ptrData;
        cv::Scalar pad_value;
        L_O = cv::Mat(img.rows, img.cols, img.type());

        std::vector<cv::Mat> imgChannles(img.channels());
        cv::split(img, imgChannles);
        std::vector<cv::Mat> L_O_channles(img.channels());
        cv::split(L_O, L_O_channles);

        if (metric_par.surround_l == -1)
            pad_value = cv::mean(img);
        else {
            pad_value[2] = pad_value[1] = pad_value[0] = metric_par.surround_l;
        }

        for(unsigned int k = 0; k < imgChannles.size(); k++) {
            if (metric_par.do_mtf) {
                L_O_channles[k] = fft_convolution(imgChannles[k], m_mtf_filter, pad_value[k]);
                ptrData = (double*) L_O_channles[k].data;

                for( int i = 0; i < L_O_channles[k].rows * L_O_channles[k].cols; ++i) {
                    if (*ptrData < 1e-5 )
                        *ptrData = 1e-5;
                    if (*ptrData > 1e10)
                        *ptrData = 1e10;

                    ptrData++;
                }
            }
            else
                L_O_channles[k] = imgChannles[k].clone();
        }
        cv::merge(L_O_channles, L_O);
    }


    void HDRVDP_VisPathway::photoSpectralSensitivity(const MetricParams &metric_par) {
        M_img_lmsr = cv::Mat(m_imgChannels, 4, CV_64FC1);
        std::vector<double> a(0);
        std::vector<double> b(0);
        std::vector<double> y(0);
        std::vector<double> result(0);

        double *lmsrPtr = (double *) LMSR_S.data;
        double *imgePtr = (double *) metric_par.spectral_emission.data;

        //GPLib::writeCvMatToFile<double>(metric_par.spectral_emission, "IMG_E.yml", true);
        //exit(0);


        for(unsigned int i = 0; i < 4; i++) {
            for(unsigned int j = 0; j < m_imgChannels; j++) {
                for(unsigned int k = 0; k < static_cast<unsigned int>(LMSR_S.rows); k++) {
                    y.push_back((*(lmsrPtr + k*4 + i)) * (*(imgePtr + k*m_imgChannels + j)));
                }

                M_img_lmsr.at<double>(j,i) = trapz(this->lambda, y);
                y.clear();
            }
        }

        L_O = L_O.reshape(1, L_O.rows * L_O.cols);
        L_O = L_O * M_img_lmsr;
        R_LMSR = L_O.reshape(4, m_imgHeight);

        std::vector<cv::Mat> R_LMSR_Channles(4);
        cv::split(R_LMSR, R_LMSR_Channles);

        L_adapt = R_LMSR_Channles[0] + R_LMSR_Channles[1];

        cv::Scalar La = cv::mean(L_adapt);
        cv::Mat P_LMR;
        std::vector<cv::Mat> P_LMR_channels(0);
        int ph_type;
        int ii, k;
        double* ptrToData;

        for(ii = 0, k = 0; k <= 3; k++) {
            if (k == 3) {
                ii = 2;
                ph_type = 1;
            }
            else {
                ii = k;
                ph_type = 0;
            }

            ptrToData = (double*) R_LMSR_Channles[k].data;
			


            double origin = pn_Y.at<double>(0);
            double increment = pn_Y.at<double>(1) - pn_Y.at<double>(0);
            std::map<double, double> X_V;

            for (int i = 0; i < pn_jnd[ph_type].cols; i++)
                X_V.insert(std::pair<double, double>(origin + increment*i, pn_jnd[ph_type].at<double>(i)));

            std::vector<double> Vq(0);


			/*std::vector<double> Xv_vec(0);
			for( int i = 0; i < R_LMSR_Channles[k].rows * R_LMSR_Channles[k].cols; ++i) {
			if (*ptrToData < pow(10.0, pn_Y.at<double>(0)))
			*ptrToData = pow(10.0, pn_Y.at<double>(0));
			if (*ptrToData > pow(10.0, pn_Y.at<double>(pn_Y.rows * pn_Y.cols - 1)))
			*ptrToData = pow(10.0, pn_Y.at<double>(pn_Y.rows * pn_Y.cols - 1));

			*ptrToData = log10(*ptrToData);
			Xv_vec.push_back(*ptrToData);

			*ptrToData++;
			}
            Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec); //this one*/

			Vq = linear_interpolation_GPU(X_V, R_LMSR_Channles[k].rows * R_LMSR_Channles[k].cols, ptrToData); //this one
            cv::Mat tmpMat(m_imgHeight, m_imgWidth, CV_64FC1);
            for(unsigned int col = 0; col < m_imgWidth; col++) {
                for(unsigned int row = 0; row < m_imgHeight; row++) {
                    tmpMat.at<double>(row, col) = Vq.at(row*m_imgWidth + col);
                }
            }

            P_LMR_channels.push_back(tmpMat);

            if (k == 1)
                k++;
        }

        cv::Mat P_C = P_LMR_channels.at(0) + P_LMR_channels.at(1);
        cv::Scalar mm = cv::mean(P_C);

        P_C = P_C - mm[0];

        mm = cv::mean(P_LMR_channels.at(2));
        cv::Mat P_R = P_LMR_channels.at(2) - mm[0];
        cv::Mat P = P_C + P_R;

// SPYR PYR

        PFILTER PF;
        //MATRIX P_Matrix = (MATRIX) malloc(sizeof(struct _matrix));
        MATRIX P_Matrix = NewMatrix(P.rows, P.cols);

        PF = LoadPFilter(metric_par.steerpyr_filter.c_str());
        if (PF == NULL)
            throw std::runtime_error("ERROR: steerpyr_filter file not found!");


        P.convertTo(P, CV_32FC1);
        P_Matrix->rows = P.rows;
        P_Matrix->columns = P.cols;
        //P_Matrix->values = (float *) P.data;

        float *pPtr = (float *) P.data;
        float *p_matPtr = P_Matrix->values;

        for (int i = 0; i < P.rows * P.cols; i++) {
            *p_matPtr = *pPtr;
            p_matPtr++; pPtr++;
        }
                
        int levels = defaultLevelsNum(P_Matrix->columns, P_Matrix->rows, PF->snd_lowband_filter->columns);

        P_pyramid = CreatePyramid(P_Matrix, PF, levels);
        /////////////////////////////////////////////////
        //MATRIX tmpMatrix = GetSubbandImage(P_pyramid, 1, 3);
        //float *tmpMatrixPtr = tmpMatrix->values;

        //if (1) {
        //    for (int i = 0; i < tmpMatrix->rows * tmpMatrix->columns; i++) {
        //        *tmpMatrixPtr = -1 * (*tmpMatrixPtr);
        //        tmpMatrixPtr++;
        //    }
        //}
        //SaveMatrix(tmpMatrix, "refBand.yml", 1);
        //std::cout << "END: PRESS ENTER...\n";
        //std::cin.ignore();
        //exit(EXIT_SUCCESS);
        ////////////////////////////////////////////////

        D_bands = CreatePyramid(P_Matrix, PF, levels);
        bandSize = levels + 2;

        band_freq = new double[bandSize];
        bands_sz = new int[bandSize];

        bands_sz[0] = bands_sz[levels + 1] = 1;
        bandsSum = 2;

        for(int i = 1; i < (levels + 1); i++) {
            bands_sz[i] = PF->num_orientations;
            bandsSum += bands_sz[i];
        }

        for(int i = 0; i < (levels + 2); i++)
            band_freq[i] = pow(2, -i) * metric_par.pix_per_degree / 2;


        cv::Mat BB(P_pyramid->lowband->rows, P_pyramid->lowband->columns,CV_32FC1, P_pyramid->lowband->values);
        BB.convertTo(BB, CV_64FC1);

        cv::Mat rho_bb = CreateCycdegImage(P_pyramid->lowband->rows, P_pyramid->lowband->columns, \
                                           band_freq[levels + 1]*2*sqrt(2));

        cv::Mat csf_bb = hdrvdp_ncsf(rho_bb, La[0], metric_par);
        cv::transpose(csf_bb, csf_bb);

        double pad_value = metric_par.surround_l == -1 ? cv::mean(BB)[0] : metric_par.surround_l;

        BB = fft_convolution(BB, csf_bb, pad_value);
        BB.convertTo(BB, CV_32FC1);

        for (int r = 0; r < BB.rows; r++) {
            for (int c = 0; c < BB.cols; c++) {
                MatrixSet(P_pyramid->lowband, r, c, BB.at<float>(r,c));
            }
        }
    }


    int HDRVDP_VisPathway::defaultLevelsNum(int imgCols, int imgRows, int filterSize) {
        int levels = 0;

        while (imgCols > filterSize && imgRows > filterSize) {
            imgCols = static_cast <int> (std::floor(imgCols / 2.0));
            imgRows = static_cast <int> (std::floor(imgRows / 2.0));

            levels++;
        }

        return levels;
    }

    cv::Mat HDRVDP_VisPathway::hdrvdp_ncsf(const cv::Mat &rho, double lum, const MetricParams &metric_par) {
        cv::Mat result(rho.rows, rho.cols, CV_64FC1);
        cv::Mat par = cv::Mat::zeros(1, 4, CV_64FC1);
        std::vector<double> Xv_vec;

        double log_lum = log10(lum);
        double *lum_lut = new double[6];
        
        for (int i = 0; i < 6; i++)
            lum_lut[i] = log10(metric_par.csf_lums[i]);
        Xv_vec.push_back(min(max(log_lum, lum_lut[0]), lum_lut[5]));
        
        std::map<double, double> X_V;
        std::vector<double> Vq;

        for(int i = 1; i < 5; i++) {
            for (int j = 0; j < 6; j++)
                X_V.insert(std::pair<double, double>(lum_lut[j], metric_par.csf_params[j][i]));
            
            par.at<double>(0, i-1) = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec)[0];
            
            X_V.clear();
        }

        double* ptrResData = (double*) result.data;
        double* ptrRhoData = (double*) rho.data;

        double par1 = par.at<double>(0,0);
        double par2 = par.at<double>(0,1);
        double par3 = par.at<double>(0,2);
        double par4 = par.at<double>(0,3);

        for( int i = 0; i < result.rows * result.cols; ++i) {
                *ptrResData++ = par4 * (1.0 / pow((1.0 + pow(par1*(*ptrRhoData), par2)) * (1.0 / pow(1.0- pow(exp((-(*ptrRhoData))/7.0), 2.0), par3)), 0.5));
                ptrRhoData++;
        }

        return result;
    }


    cv::Mat HDRVDP_VisPathway::hdrvdp_ncsf(double rho, const cv::Mat &lum, const MetricParams &metric_par) {
        cv::Mat result(lum.rows, lum.cols, CV_64FC1);
        cv::Mat par = cv::Mat::zeros(lum.rows, 4, CV_64FC1);
        std::vector<double> Xv_vec;

        cv::Mat log_lum(lum.rows, lum.cols, lum.type());
        HDRVDP_helper::cvMatLog10(lum, log_lum);

        double *lum_lut = new double[6];        
        for (int i = 0; i < 6; i++)
            lum_lut[i] = log10(metric_par.csf_lums[i]);
        

        double *ptrLogLum = (double *) log_lum.data;
        for (int i = 0; i < log_lum.cols * log_lum.rows; i++)
            Xv_vec.push_back(min(max(*ptrLogLum++, lum_lut[0]), lum_lut[5]));
        
        std::map<double, double> X_V;
        std::vector<double> Vq;
        std::vector<double> testVec;

        for(int i = 1; i < 5; i++) {
            for (int j = 0; j < 6; j++)
                X_V.insert(std::pair<double, double>(lum_lut[j], metric_par.csf_params[j][i]));
            
            testVec = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec);
            for(unsigned int k = 0; k < testVec.size(); k++)
                par.at<double>(k, i -1) = testVec[k];

            X_V.clear();
        }

        double par1, par2, par3, par4;

        for( int r = 0; r < result.rows; ++r) {
            for (int c = 0; c < result.cols; ++c) {
                par1 = par.at<double>(r,0);
                par2 = par.at<double>(r,1);
                par3 = par.at<double>(r,2);
                par4 = par.at<double>(r,3);

                result.at<double>(r,c) = par4 * (1.0 / pow((1.0 + pow(par1*rho, par2)) * (1.0 / pow(1.0- pow(exp((-(rho))/7.0), 2.0), par3)), 0.5));
            }
        }

        return result;
    }

    cv::Mat HDRVDP_VisPathway::fft_convolution(const cv::Mat &image, const cv::Mat &filter, double padValue) {
        int M = filter.rows;
        int N = filter.cols;

        if ((M & (M - 1)) == 0 )
            M = GPLib::roundUp(M, 2);
        else if ((N & (N - 1)) == 0 )
            N = GPLib::roundUp(N, 2);

        cv::Mat tmpImage, tmpFilter;
        cv::copyMakeBorder(image, tmpImage, 0, M - image.rows, 0, N - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(padValue));
        cv::copyMakeBorder(filter, tmpFilter, 0, M - filter.rows, 0, N - filter.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        size_t numOfPixels = M * N;

        // allocate input arrays
        fftw_complex *in_image  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);

        // assign values to real parts (values between 0 and MaxRGB)
        for(unsigned int i = 0; i < numOfPixels; i++) {
            // save as real numbers
            in_image[i][0]  = *((double *) tmpImage.data + i);// / 255.0;
        }

        // allocate output arrays
        fftw_complex *out_image  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * numOfPixels);

        // create FTT plans
        fftw_plan ftt_image  = fftw_plan_dft_2d(M, N, in_image, out_image, FFTW_FORWARD, FFTW_ESTIMATE);

        // perform FORWARD fft
        fftw_execute(ftt_image);

        fftw_destroy_plan(ftt_image);

        // transform imaginary number to phase and magnitude and save to output
        for(unsigned int i = 0; i < numOfPixels; i++) {
            double kernelVal = *((double *) filter.data + i);

            in_image[i][0] = out_image[i][0] * kernelVal;
            in_image[i][1] = out_image[i][1] * kernelVal;
        }

        // create IFTT plans
        fftw_plan ifft_image = fftw_plan_dft_2d(M, N, in_image, out_image, FFTW_BACKWARD, FFTW_ESTIMATE);

        // perform ifft
        fftw_execute(ifft_image);

        fftw_destroy_plan(ifft_image);

        fftw_free(in_image);

        // save real parts to output
        cv::Mat result = cv::Mat::zeros(M, N, CV_64FC1);
        double scale = 1.0 / numOfPixels;
        for(unsigned int i = 0; i < numOfPixels; i++) {
            *((double *) result.data + i) = out_image[i][0] * scale;
        }

        fftw_free(out_image);

        result.adjustROI(0, -(M - image.rows), 0, -(N - image.cols));

        return result;
    }


    void HDRVDP_VisPathway::fastElementWiseMul(const cv::Mat &in1, const cv::Mat &in2, cv::Mat &dst) {
        double* ptrIn1, *ptrIn2, *ptrDst;
        int numOfElements = in1.rows * in1.cols;        

        for (int channel = 0; channel < in1.channels(); channel++) {
            ptrIn1 = (double *) (in1.data);
            ptrDst = (double *) (dst.data);
            ptrIn2 = (double*) in2.data;

            for( int i = 0; i < numOfElements; ++i) {
                *(ptrDst + ((i*in1.channels()) + channel)) = *(ptrIn1 + ((i*in1.channels()) + channel)) * (*ptrIn2++);
            }
        }

    }


    void HDRVDP_VisPathway::initLambda() {
        double tmp_lambda[420] = {360.0,361.0,362,363.01,364.01,365.01,366.01,367.02,368.02,369.02,370.02,371.03,\
                    372.03,373.03,374.03,375.04,376.04,377.04,378.04,379.05,380.05,381.05,382.05,383.05,\
                    384.06,385.06,386.06,387.06,388.07,389.07,390.07,391.07,392.08,393.08,394.08,395.08, \
                    396.09,397.09,398.09,399.09,400.1,401.1,402.1,403.1,404.11,405.11,406.11,407.11,408.11,\
                    409.12,410.12,411.12,412.12,413.13,414.13,415.13,416.13,417.14,418.14,419.14,420.14, \
                    421.15,422.15,423.15,424.15,425.16,426.16,427.16,428.16,429.16,430.17,431.17,432.17,\
                    433.17,434.18,435.18,436.18,437.18,438.19,439.19,440.19,441.19,442.2,443.2,444.2,445.2,\
                    446.21,447.21,448.21,449.21,450.21,451.22,452.22,453.22,454.22,455.23,456.23,457.23,\
                    458.23,459.24,460.24,461.24,462.24,463.25,464.25,465.25,466.25,467.26,468.26,469.26,\
                    470.26,471.26,472.27,473.27,474.27,475.27,476.28,477.28,478.28,479.28,480.29,481.29,\
                    482.29,483.29,484.3,485.3,486.3,487.3,488.31,489.31,490.31,491.31,492.32,493.32,494.32,\
                    495.32,496.32,497.33,498.33,499.33,500.33,501.34,502.34,503.34,504.34,505.35,506.35,\
                    507.35,508.35,509.36,510.36,511.36,512.36,513.37,514.37,515.37,516.37,517.37,518.38,\
                    519.38,520.38,521.38,522.39,523.39,524.39,525.39,526.4,527.4,528.4,529.4,530.41,531.41,\
                    532.41,533.41,534.42,535.42,536.42,537.42,538.42,539.43,540.43,541.43,542.43,543.44,\
                    544.44,545.44,546.44,547.45,548.45,549.45,550.45,551.46,552.46,553.46,554.46,555.47,\
                    556.47,557.47,558.47,559.47,560.48,561.48,562.48,563.48,564.49,565.49,566.49,567.49,\
                    568.5,569.5,570.5,571.5,572.51,573.51,574.51,575.51,576.52,577.52,578.52,579.52,580.53,\
                    581.53,582.53,583.53,584.53,585.54,586.54,587.54,588.54,589.55,590.55,591.55,592.55,\
                    593.56,594.56,595.56,596.56,597.57,598.57,599.57,600.57,601.58,602.58,603.58,604.58,\
                    605.58,606.59,607.59,608.59,609.59,610.6,611.6,612.6,613.6,614.61,615.61,616.61,617.61,\
                    618.62,619.62,620.62,621.62,622.63,623.63,624.63,625.63,626.63,627.64,628.64,629.64,\
                    630.64,631.65,632.65,633.65,634.65,635.66,636.66,637.66,638.66,639.67,640.67,641.67,\
                    642.67,643.68,644.68,645.68,646.68,647.68,648.69,649.69,650.69,651.69,652.7,653.7,654.7,\
                    655.7,656.71,657.71,658.71,659.71,660.72,661.72,662.72,663.72,664.73,665.73,666.73,667.73,\
                    668.74,669.74,670.74,671.74,672.74,673.75,674.75,675.75,676.75,677.76,678.76,679.76,\
                    680.76,681.77,682.77,683.77,684.77,685.78,686.78,687.78,688.78,689.79,690.79,691.79,692.79,\
                    693.79,694.8,695.8,696.8,697.8,698.81,699.81,700.81,701.81,702.82,703.82,704.82,705.82,\
                    706.83,707.83,708.83,709.83,710.84,711.84,712.84,713.84,714.84,715.85,716.85,717.85,718.85,\
                    719.86,720.86,721.86,722.86,723.87,724.87,725.87,726.87,727.88,728.88,729.88,730.88,731.89,\
                    732.89,733.89,734.89,735.89,736.9,737.9,738.9,739.9,740.91,741.91,742.91,743.91,744.92,745.92,\
                    746.92,747.92,748.93,749.93,750.93,751.93,752.94,753.94,754.94,755.94,756.95,757.95,758.95,759.95,\
                    760.95,761.96,762.96,763.96,764.96,765.97,766.97,767.97,768.97,769.98,770.98,771.98,772.98,773.99,\
                    774.99,775.99,776.99,778.0,779.0,780.0};

        for(int i = 0; i < 420; i++ )
            this->lambda.push_back(tmp_lambda[i]);

        return;
    }


    double HDRVDP_VisPathway::trapz(std::vector<double> &x, std::vector<double> &y) {
        double h = x[1] - x[0];
        int k = 0;
        double sum = 0.0;

        for(unsigned int i=0; i < x.size() - 1;i++) {
            if(k==0) {
                sum = sum + y[i];
                k=1;
            }
            else
                sum = sum + 2 * y[i];
        }

        sum = sum + y[y.size() - 1];
        sum = sum * (h/2);

        return sum;
        //double h,s;

        //h = (x.back() - x.front())/x.size();
        //s = y.front() + y.back();

        //for(int i = 1; i < x.size(); i++)
        //    s += 2.0 * y.at(i);

        //return (h/2.0)*s;
    }
}
