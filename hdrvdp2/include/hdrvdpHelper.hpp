#ifndef DIP_HDRVDP_HELPER_HPP
#define DIP_HDRVDP_HELPER_HPP

#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>

namespace dip {
    enum COLOR_ENC {
        SRGB_DISPLAY,
        RGB_BT_709,
        LUMINANCE,
        LUMA_DISPLAY
    };

    enum SPACE_TYPE {
        LINEAR,
        LOG
    };

    class HDRVDP_helper {
    public:
        static double pix_per_deg(float display_diagonal_INCH, cv::Size resolution, float viewing_distance_M);
        static cv::Mat createSpace(double lmin, double lmax, unsigned int N, SPACE_TYPE type);
        static std::vector<double> linear_interpolation_CPU(std::map<double, double> &X_V, std::vector<double> &Xv);
        static void cvMatLog10(const cv::Mat &src, cv::Mat &dst);
        static void cvMatClamp(cv::Mat &src, double minVal, double maxVal);
        static void cvMatPerElementMul(const cv::Mat &mat1, const cv::Mat &mat2, cv::Mat &dst);
        static void cvMatPerElementDiv(const cv::Mat &mat1, const cv::Mat &mat2, cv::Mat &dst);
        static double msre(const cv::Mat &mat);
        
        template <typename T> static T clamp(const T& n, const T& lower, const T& upper) {
            return std::max(lower, std::min(n, upper));
        }

        static void signPow(const cv::Mat &src, cv::Mat &dst, double e);
        static cv::Mat cumsum(const cv::Mat &src);

    private:
        HDRVDP_helper();
        template <typename T> static T sgn(T val) {
            return (T(0) < val) - (val < T(0));
        }

    };
}

#endif