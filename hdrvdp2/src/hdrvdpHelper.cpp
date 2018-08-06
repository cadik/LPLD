#define PI 3.14159265358979323846

#include "hdrvdpHelper.hpp"
#include <iostream>
#include <cmath>

namespace dip {

    double HDRVDP_helper::pix_per_deg(float display_diagonal_INCH, cv::Size resolution, float viewing_distance_M) {
        double ar = resolution.width / resolution.height;
        double height_mm = sqrt(pow((display_diagonal_INCH*25.4),2) / (1+pow(ar,2)));
        double height_deg = 2 * ((atan( 0.5*height_mm/(viewing_distance_M*1000) )*180)/PI);

        return resolution.height / height_deg;
    }


	cv::Mat HDRVDP_helper::createSpace(double d1, double d2, unsigned int N, SPACE_TYPE type) {
		double n1 = floor(N) - 1;
		std::vector<double> valuesVec(0);
        double value;

        for (int i = 0; i <= n1; i++) {
            value = d1 + i * (d2 - d1)/n1;
            valuesVec.push_back(value);
        }

        valuesVec.at(0) = d1;
        valuesVec.at(valuesVec.size()-1) = d2;

        cv::Mat out(1, valuesVec.size(), CV_64FC1);
        for(int i=0; i < out.cols; ++i) {
            if (type == LOG)
                out.at<double>(0, i) = pow(10.0, valuesVec.at(i));
            else
                out.at<double>(0, i) = valuesVec.at(i);
        }

        return out;
    }


    std::vector<double> HDRVDP_helper::linear_interpolation_CPU(std::map<double, double> &X_V, std::vector<double> &Xv) {
        std::vector<double> Vq;
        
        typedef std::map<double, double>::const_iterator i_t;

        for (unsigned int j = 0; j < Xv.size(); j++) {
            i_t i=X_V.upper_bound(Xv.at(j));
            if(i==X_V.end()) {
                 Vq.push_back((--i)->second);
                 continue;
            }

            if (i==X_V.begin()) {
                 Vq.push_back(i->second);
                 continue;
            }
            
            i_t l=i;
            --l;

            const double delta =(Xv.at(j) - l->first)/(i->first - l->first);
            Vq.push_back(delta*i->second +(1-delta)*l->second);
        }

        return Vq;
    }

    void HDRVDP_helper::cvMatLog10(const cv::Mat &src, cv::Mat &dst) {
        double* ptrSRC = (double*) src.data;
        double* ptrDST = (double*) dst.data;

        for( int i = 0; i < src.rows * src.cols; ++i)
            *ptrDST++ = log10(*ptrSRC++);
    }

    void HDRVDP_helper::cvMatClamp(cv::Mat &src, double minVal, double maxVal) {
        double *srcPtr = (double *) src.data;

        for (int i = 0; i < src.rows * src.cols * src.channels(); i++) {
            *srcPtr = cv::min(cv::max(*srcPtr, minVal), maxVal);
            srcPtr++;
        }
    }

    void HDRVDP_helper::signPow(const cv::Mat &src, cv::Mat &dst, double e) {
        double *srcPtr = (double *) src.data;
        double *dstPtr = (double *) dst.data;

        for (int i = 0; i < src.rows * src.cols * src.channels(); i++) {
	    *dstPtr++ = sgn<double>(*srcPtr) * pow(std::abs(*srcPtr), e);
            srcPtr++;
        }
    }


	cv::Mat HDRVDP_helper::cumsum(const cv::Mat &src) {
		cv::Mat result(src.rows, src.cols, CV_64FC1);
		double sum = 0.0;
		double *srcPtr = (double *)src.data;
		double *resPtr = (double *)result.data;
		
		for (int i = 0; i < src.rows * src.cols; i++) {
			sum += *srcPtr++;
			*resPtr++ = sum;
		}

		return result;
	}

    void HDRVDP_helper::cvMatPerElementMul(const cv::Mat &mat1, const cv::Mat &mat2, cv::Mat &dst) {
        double *mat1Ptr = (double *) mat1.data;
        double *mat2Ptr = (double *) mat2.data;
        double *dstPtr = (double *) dst.data;

        for (int i = 0; i < mat1.rows * mat1.cols * mat1.channels(); i++) {
            *dstPtr++ = (*mat1Ptr++) * (*mat2Ptr++);
        }
    }

    void HDRVDP_helper::cvMatPerElementDiv(const cv::Mat &mat1, const cv::Mat &mat2, cv::Mat &dst) {
        double *mat1Ptr = (double *)mat1.data;
        double *mat2Ptr = (double *)mat2.data;
        double *dstPtr = (double *)dst.data;

        for (int i = 0; i < mat1.rows * mat1.cols * mat1.channels(); i++) {
            *dstPtr++ = (*mat1Ptr++) / (*mat2Ptr++);
        }
    }


    double HDRVDP_helper::msre(const cv::Mat &mat) {
        double sum = 0.0;
        int numOfElements = mat.rows * mat.cols * mat.channels();
        double *matPtr = (double *) mat.data;

        for(int i = 0; i < numOfElements; i++)
            sum = sum + pow(*matPtr++, 2.0);

        return sqrt(sum) / numOfElements;
    }
}