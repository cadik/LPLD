#include "hdrvdpVisualize.hpp"

namespace dip {
    double HDRVD_Visualize::color_map_data[5][3] = { { 0.2, 0.2, 1.0 }, \
                                                     { 0.2, 1.0, 1.0 }, \
                                                     { 0.2, 1.0, 0.2 }, \
                                                     { 1.0, 1.0, 0.2 }, \
                                                     { 1.0, 0.2, 0.2 } };

    double HDRVD_Visualize::color_map_in_data[5] = { 0.0, 0.25, 0.5, 0.75, 1 };
    double HDRVD_Visualize::screenValsMul_data[3] = { 0.2126, 0.7152, 0.0722 };

    HDRVD_Visualize::HDRVD_Visualize(const cv::Mat &P, const cv::Mat &img) {
        cv::Mat tmo_img = vis_tonemap(log_luminance(img), 0.6);

        cv::Mat color_map(5, 3, CV_64FC1, &color_map_data);
        cv::Mat color_map_in(1, 5, CV_64FC1, &color_map_in_data);
        cv::Mat screenValsMulMat(1, 3, CV_64FC1, &screenValsMul_data);
        cv::transpose(screenValsMulMat, screenValsMulMat);
        
        cv::Mat color_map_l = color_map * screenValsMulMat;
        color_map_l = cv::repeat(color_map_l, 1, 3);

        cv::Mat color_map_ch(color_map.rows, color_map.cols, color_map.type());
        HDRVDP_helper::cvMatPerElementDiv(color_map, color_map_l, color_map_ch);

        std::vector<cv::Mat> channels;

        std::map<double, double> X_V;
        std::vector<double> Vq;
        std::vector<double> Xq;

        double * pPtr = (double *)P.data;
        for (int i = 0; i < P.rows * P.cols; i++)
            Xq.push_back(*pPtr++);

        double *cMapInPtr;

        for (int i = 0; i < 3; i++) {
            cMapInPtr = (double *)color_map_in.data;

            for (int j = 0; j < color_map_in.cols; j++)
                X_V.insert(std::pair<double, double>(*cMapInPtr++, color_map_ch.at<double>(j, i)));

            cv::Mat channelTmpMat = cv::Mat::zeros(P.rows, P.cols, CV_64FC1);

            Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xq);
            memcpy(channelTmpMat.data, Vq.data(), Vq.size()*sizeof(double));

            channels.push_back(channelTmpMat);

            X_V.clear(); std::map<double, double>().swap(X_V);
            Vq.clear(); std::vector<double>().swap(Vq);
        }

        Xq.clear(); std::vector<double>().swap(Xq);

        map = cv::Mat::zeros(img.rows, img.cols, CV_64FC3);
        cv::merge(channels, map);

        channels.clear(); std::vector<cv::Mat>().swap(channels);
        channels.push_back(tmo_img);
        channels.push_back(tmo_img);
        channels.push_back(tmo_img);

        cv::merge(channels, tmo_img);

        HDRVDP_helper::cvMatPerElementMul(map, tmo_img, map);

        map.convertTo(map, CV_32FC3);
        cv::cvtColor(map, map, CV_RGB2BGR);
    }


    cv::Mat HDRVD_Visualize::log_luminance(const cv::Mat &X) const {
        cv::Mat Y(X.rows, X.cols, CV_64FC1);

        double *yPtr;
        double minValue = 1000000.0;

        if (X.channels() == 3) {
            double *xPtr = (double *)X.data;
            yPtr = (double *)Y.data;

            double tmpVal;
            double r, g, b;

            for (int i = 0; i < X.rows * X.cols; i++) {
                r = *xPtr++ * 0.212656; g = *xPtr++ * 0.715158; b = *xPtr++ * 0.072186;

                tmpVal = r + g + b;
                if (tmpVal <= 0)
                    *yPtr++ = 0.0;
                else {
                    *yPtr++ = tmpVal;
                    if (tmpVal < minValue)
                        minValue = tmpVal;
                }
            }
        }
        else {
            X.copyTo(Y);
            cv::threshold(Y, Y, 0.0, 0.0, CV_THRESH_TOZERO);
        }


        yPtr = (double *)Y.data;

        for (int i = 0; i < Y.rows * Y.cols; i++) {
            if (*yPtr == 0.0)
                *yPtr = minValue;
            yPtr++;
        }

        cv::log(Y, Y);
        return Y;
    }


    cv::Mat HDRVD_Visualize::vis_tonemap(const cv::Mat &b, double dr) {
        double t = 3.0;
        cv::Mat tmo_img(b.rows, b.cols, CV_64FC1);

        double min, max;
        cv::minMaxLoc(b, &min, &max);
        cv::Mat bscale = HDRVDP_helper::createSpace(min, max, 1024, LINEAR);

        cv::Mat b_p;
        cv::Mat test(b.rows, b.cols, CV_32FC1);
        b.convertTo(test, CV_32FC1);

        int bins = 1024;
        int histSize[] = { 1024 };

        bscale.convertTo(bscale, CV_32FC1);
        float hranges[] = { static_cast<float>(min), static_cast<float>(max) };
        //float *ptrT = (float *)bscale.data + 1023;
        //for (int i = 0; i < 1024; i++)
        //	hranges[i] =*ptrT--;

        const float* ranges[] = { hranges };
        int channels[] = { 0 };

        bool uniform = true; bool accumulate = true;
        cv::calcHist(&test, 1, channels, cv::Mat(), b_p, 1, histSize, ranges, uniform, accumulate);

        b_p.convertTo(b_p, CV_64FC1);

        b_p = b_p / cv::sum(b_p)[0];
        cv::Mat pow_b_p;
        cv::pow(b_p, 1 / t, pow_b_p);

        cv::Mat dy = pow_b_p / cv::sum(pow_b_p)[0];
        cv::Mat v = HDRVDP_helper::cumsum(dy) * dr + (1 - dr) / 2.0;

        std::map<double, double> X_V;
        std::vector<double> Xq;
        std::vector<double> Vq;

        bscale.convertTo(bscale, CV_64FC1);

        double *bscalePtr = (double *)bscale.data;
        double *vPtr = (double *)v.data;
        for (int i = 0; i < bscale.cols; i++) {
            X_V.insert(std::pair<double, double>(*bscalePtr++, *vPtr++));
        }

        double *bPtr = (double *)b.data;
        for (int i = 0; i < b.rows * b.cols; i++)
            Xq.push_back(*bPtr++);


        Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xq);
        memcpy(tmo_img.data, Vq.data(), Vq.size()*sizeof(double));

        X_V.clear(); std::map<double, double>().swap(X_V);
        Xq.clear(); std::vector<double>().swap(Xq);
        Vq.clear(); std::vector<double>().swap(Vq);

        return tmo_img;
    }
}
