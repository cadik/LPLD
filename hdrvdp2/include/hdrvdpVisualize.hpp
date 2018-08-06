#ifndef DIP_HDRVDP_VISUALIZE_HPP
#define DIP_HDRVDP_VISUALIZE_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "hdrvdpHelper.hpp"

namespace dip {
	class HDRVD_Visualize
	{
	public:
		HDRVD_Visualize(const cv::Mat &P, const cv::Mat &img);
        cv::Mat getMap() { return map; };

	private:
        cv::Mat map;

		cv::Mat log_luminance(const cv::Mat &X) const;
		cv::Mat vis_tonemap(const cv::Mat &b, double dr);

		static double color_map_data[5][3];
        static double color_map_in_data[5];
        static double screenValsMul_data[3];
	};
}

#endif
