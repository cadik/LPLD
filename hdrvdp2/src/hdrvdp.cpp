//#include <fstream>
#include <cmath>
//#include <iomanip>
#include "hdrvdp.hpp"


namespace dip {
    void HDRVDP::compute(cv::Mat reference_img, cv::Mat test_img, COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error) {
        m_refMat = reference_img;
        m_testMat = test_img;
        
        runComputation(color_encoding, pix_per_deg);
    }


    void HDRVDP::compute(std::string reference_img, std::string test_img, \
                         COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error) {
        
        loadImages(reference_img, test_img, color_encoding);
        runComputation(color_encoding, pix_per_deg);
    }


    std::vector<std::vector<cv::Mat>> HDRVDP::getComputedBands(cv::Mat reference_img, cv::Mat test_img, COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error) {
        m_refMat = reference_img;
        m_testMat = test_img;
        
        m_color_encoding = color_encoding;
        m_metricPar.pix_per_degree = pix_per_deg;
        
        setSources();
        load_spectral_resp();
        
        m_visualPathway_testImg = new HDRVDP_VisPathway(m_testMat, m_metricPar, clObj);
        m_visualPathway_refImg = new HDRVDP_VisPathway(m_refMat, m_metricPar, clObj);

        std::vector<std::vector<cv::Mat>> result;
        std::vector<cv::Mat> band_level_diff;

        for (int b = 0; b < m_visualPathway_testImg->bandSize; b++) {
            for(int o = 0; o < m_visualPathway_testImg->bands_sz[b]; o++) {
                cv::Mat testBand = get_band(m_visualPathway_testImg->P_pyramid, b, o);
                cv::Mat refBand = get_band(m_visualPathway_refImg->P_pyramid, b, o);

                cv::Mat band_diff = (testBand - refBand);

                /*GPLib::writeCvMatToFile<double>(testBand, "result.yml", true, false);*/

                band_level_diff.push_back(band_diff);
            }
            result.push_back(band_level_diff);
            band_level_diff.clear();
        }

        return result;
    }

    void HDRVDP::runComputation(COLOR_ENC color_encoding, double pix_per_deg) throw (std::runtime_error, cl::Error) {
        m_color_encoding = color_encoding;
        m_metricPar.pix_per_degree = pix_per_deg;
        
        setSources();
        load_spectral_resp();

        m_visualPathway_testImg = new HDRVDP_VisPathway(m_testMat, m_metricPar, clObj);
        m_visualPathway_refImg = new HDRVDP_VisPathway(m_refMat, m_metricPar, clObj);

        cv::Mat L_adapt = (m_visualPathway_testImg->L_adapt + m_visualPathway_refImg->L_adapt) / 2.0;
      
        cv::Mat csf_la = HDRVDP_helper::createSpace(-5.0, 5.0, 256, LOG);
        cv::Mat csf_laT;
        cv::transpose(csf_la, csf_laT);
        
        cv::Mat csf_log_la(csf_la.rows, csf_la.cols, csf_la.type());

        HDRVDP_helper::cvMatLog10(csf_la, csf_log_la);

        cv::Mat CSF = cv::Mat::zeros(csf_la.cols, m_visualPathway_testImg->bandSize, CV_64FC1);

        /////////////// -------------- TODO!
        //PYRAMID D_bands = (PYRAMID) malloc(sizeof(PYRAMID));
        //copyPyramid(m_visualPathway_testImg->P_pyramid, D_bands);

        cv::Mat tmpMat;
        for (int b = 0; b < m_visualPathway_testImg->bandSize; b++ ) {
            tmpMat = HDRVDP_VisPathway::hdrvdp_ncsf(m_visualPathway_testImg->band_freq[b], csf_laT, m_metricPar);
            for (int i = 0; i < tmpMat.rows; i++)
                CSF.at<double>(i, b) = tmpMat.at<double>(i, 0);
        }

        HDRVDP_helper::cvMatClamp(L_adapt, csf_la.at<double>(0,0), csf_la.at<double>(csf_la.rows-1, csf_la.cols-1));
        cv::Mat log_La(L_adapt.rows, L_adapt.cols, L_adapt.type());
        HDRVDP_helper::cvMatLog10(L_adapt, log_La);

        std::vector<cv::Mat> refImgChannels(3);
        std::vector<cv::Mat> testImgChannels(3);

        cv::split(m_refMat, refImgChannels);
        cv::split(m_testMat, testImgChannels);

        cv::Mat diffMask(m_refMat.rows, m_refMat.cols, CV_64FC1);

        double *refPtr = (double *) refImgChannels[2].data;
        double *testPtr = (double *) testImgChannels[2].data;
        double *diffPtr = (double *) diffMask.data;

        for(int i = 0; i < diffMask.rows * diffMask.cols; i++) {
            *diffPtr = std::abs((*testPtr) - (*refPtr)) / (*refPtr) > 0.001 ? 1.0 : 0.0;
            refPtr++; testPtr++; diffPtr++;
        }

        double p = pow(10.0, m_metricPar.mask_p);
        double q = pow(10.0, m_metricPar.mask_q);
        double pf = pow(10.0, m_metricPar.psych_func_slope) / p;
        cv::Mat D(1, 1, CV_64FC1);
        Q = 0.0;

        MATRIX tmpMatrix;
        for (int b = 0; b < m_visualPathway_testImg->bandSize; b++) {
            if (b == 7) 
                int a = 10;
            if (b == 0)
                tmpMatrix = m_visualPathway_testImg->P_pyramid->hiband;
            else if (b == m_visualPathway_testImg->bandSize - 1)
                tmpMatrix = m_visualPathway_testImg->P_pyramid->lowband;
            else
                tmpMatrix = m_visualPathway_testImg->P_pyramid->levels[b - 1]->subband[0];

            //cv::Mat mask_xo = get_band( m_visualPathway_testImg->P_pyramid, b, 0);

            cv::Mat mask_xo = cv::Mat::zeros(tmpMatrix->rows, tmpMatrix->columns, CV_64FC1);

            for(int o = 0; o < m_visualPathway_testImg->bands_sz[b]; o++) {
                cv::Mat testBand = get_band(m_visualPathway_testImg->P_pyramid, b, o);
                cv::Mat refBand = get_band(m_visualPathway_refImg->P_pyramid, b, o);

                mask_xo = mask_xo + mutual_masking(testBand, refBand);
            }

            cv::Mat log_La_rs;
            cv::resize(log_La, log_La_rs, cv::Size(mask_xo.cols, mask_xo.rows), 0, 0, CV_INTER_LINEAR);

            double mini = csf_log_la.at<double>(0,0);
            double maxi =  csf_log_la.at<double>(csf_log_la.rows -1, csf_log_la.cols-1);

            HDRVDP_helper::cvMatClamp(log_La_rs, mini, maxi);

            std::map<double, double> X_V;
            std::vector<double> Xv_vec(0);
            std::vector<double> Vq(0);

            for (int i = 0; i < CSF.rows; i++)
                 X_V.insert(std::pair<double, double>(csf_log_la.at<double>(0, i), CSF.at<double>(i, b)));
            
            double *log_laPtr = (double *) log_La_rs.data;
            for(int i = 0; i < log_La_rs.rows * log_La_rs.cols; i++)
                Xv_vec.push_back(*log_laPtr++);

            Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec);

            cv::Mat CSF_b(log_La_rs.rows, log_La_rs.cols, CV_64FC1);
            memcpy(CSF_b.data,Vq.data(),Vq.size()*sizeof(double));

            X_V.clear(); std::map<double, double>().swap(X_V);
            Xv_vec.clear(); std::vector<double>().swap(Xv_vec);
            Vq.clear(); std::vector<double>().swap(Vq);

            double band_norm = pow(2.0, b);
            double band_mul = 1.0;

            for(int o = 0; o < m_visualPathway_testImg->bands_sz[b]; o++) {
                cv::Mat testBand = get_band(m_visualPathway_testImg->P_pyramid, b, o);
                cv::Mat refBand = get_band(m_visualPathway_refImg->P_pyramid, b, o);
        
                cv::Mat band_diff = (testBand - refBand) * band_mul;
//--------------------------------------------------------------------------------------
        //std::vector<cv::Mat> P_LMR_channels;
        //cv::split(P_LMR, P_LMR_channels);

        GPLib::writeCvMatToFile<double>(band_diff, "band_diff.yml", false, false);

        std::cout << "Press enter...\n";
//        std::cin.ignore();
        //exit(EXIT_SUCCESS);
//--------------------------------------------------------------------------------------

                cv::Mat ex_diff(band_diff.rows, band_diff.cols, CV_64FC1);

                HDRVDP_helper::signPow(band_diff / band_norm, ex_diff, p);
        
                ex_diff = ex_diff * band_norm;

                cv::Mat N_nCSF; 
                if (b != m_visualPathway_testImg->bandSize - 1)
                    N_nCSF = 1 / CSF_b;

                if (m_metricPar.do_masking) {
                    double k_mask_self = pow(10.0, m_metricPar.mask_self);
                    double k_mask_xo = pow(10.0, m_metricPar.mask_xo);            
                    double k_mask_xn = pow(10.0, m_metricPar.mask_xn);

                    cv::Mat self_mask = mutual_masking(testBand, refBand);
                    cv::Mat mask_xn = cv::Mat::zeros(self_mask.rows, self_mask.cols, CV_64FC1);

                    if ( b > 0) {
                        cv::Mat testBand = get_band(m_visualPathway_testImg->P_pyramid, b-1, o);
                        cv::Mat refBand = get_band(m_visualPathway_refImg->P_pyramid, b-1, o);
                        cv::Mat tmpMat;
                        //mask_xn = max( imresize( mutual_masking( b-1, o ), size( self_mask ) ), 0 )/(band_norm/2);
                        cv::resize(mutual_masking(testBand, refBand), tmpMat, cv::Size(self_mask.cols, self_mask.rows), 0, 0, CV_INTER_LINEAR);
                        mask_xn = cv::max(tmpMat, 0.0) / (band_norm/2.0);
                    }
                    if (b < m_visualPathway_testImg->bandSize - 1) {
                        cv::Mat testBand = get_band(m_visualPathway_testImg->P_pyramid, b+1, o);
                        cv::Mat refBand = get_band(m_visualPathway_refImg->P_pyramid, b+1, o);

                        cv::Mat tmpMat;// = mutual_masking(testBand, refBand);
                        cv::resize(mutual_masking(testBand, refBand), tmpMat, cv::Size(self_mask.cols, self_mask.rows), 0, 0, CV_INTER_LINEAR);

                        mask_xn = mask_xn + cv::max(tmpMat, 0.0) / (band_norm*2.0);
                    }

                    cv::Mat band_mask_xo = cv::max(mask_xo - self_mask, 0.0);

                    cv::Mat N_mask(band_mask_xo.rows, band_mask_xo.cols, CV_64FC1);
                    cv::resize(D, D, cv::Size(band_mask_xo.cols, band_mask_xo.rows));
                    
                    if (b == m_visualPathway_testImg->bandSize - 1)
                        N_nCSF = cv::Mat::ones(D.rows, D.cols, CV_64FC1);

                    double *n_maskPtr, *dPtr, *ex_diffPtr;
                    double *selfM_Ptr, *n_ncsfPtr, *bandM_xoPtr, *mask_xnPtr;

                    n_maskPtr = (double *) N_mask.data;
                    dPtr = (double *) D.data;
                    selfM_Ptr = (double *) self_mask.data;
                    n_ncsfPtr = (double *) N_nCSF.data;
                    bandM_xoPtr = (double *) band_mask_xo.data;
                    mask_xnPtr = (double *) mask_xn.data;
                    ex_diffPtr = (double *) ex_diff.data;

                    for (int i = 0; i < N_mask.rows * N_mask.cols; i++) {
                        *n_maskPtr = band_norm * ((k_mask_self * pow(((*selfM_Ptr++) /(*n_ncsfPtr)) /band_norm, q)) + \
                                           (k_mask_xo   * pow(((*bandM_xoPtr++) /(*n_ncsfPtr)) /band_norm, q)) + \
                                           (k_mask_xn   * pow(((*mask_xnPtr++)  /(*n_ncsfPtr)), q)));

                        *dPtr++ = *ex_diffPtr++ / sqrt(pow(*n_ncsfPtr++, 2.0*p) + pow(*n_maskPtr++, 2.0));
                    }
                    
                    cv::Mat A(D.rows, D.cols, D.type());
                    HDRVDP_helper::signPow(D/band_norm, A, pf);
                    cv::Mat B(D.rows, D.cols, D.type());

                    B = A * band_norm;
                    if (o == 0 && b > 0 && b != 7)
                        B = B * (-1.0);

                    set_band(m_visualPathway_testImg->D_bands, b, o, B);
                }
                else {
                    // !TODO: NO MASKING !!!
                }


                std::map<double, double> X_V;
                std::vector<double> Xv_vec(0);
                std::vector<double> Vq(0);

                for (int i = 0; i < 7; i++)
                     X_V.insert(std::pair<double, double>(m_metricPar.quality_band_freq[i], m_metricPar.quality_band_w[i]));
            
                Xv_vec.push_back(HDRVDP_helper::clamp(m_visualPathway_testImg->band_freq[b], m_metricPar.quality_band_freq[6], m_metricPar.quality_band_freq[0]));

                Vq = HDRVDP_helper::linear_interpolation_CPU(X_V, Xv_vec);

                double w_f = Vq.at(0);
                double epsilon = 0.00001;

                cv::Mat diffMask_b;
                cv::resize(diffMask, diffMask_b, cv::Size(D.cols, D.rows), 0, 0, CV_INTER_LINEAR);
                cv::Mat D_p(D.rows, D.cols, D.type());

                HDRVDP_helper::cvMatPerElementMul(D, diffMask_b, D_p);

                Q = Q + log(HDRVDP_helper::msre(D_p) + epsilon) * w_f / m_visualPathway_testImg->bandsSum;
            }
        }

        MATRIX S_mapMat = CollapsePyramid(m_visualPathway_testImg->D_bands);
        
        S_map = cv::Mat(S_mapMat->rows, S_mapMat->columns, CV_64FC1);
        P_map = cv::Mat(S_mapMat->rows, S_mapMat->columns, CV_64FC1);

        float *s_matrixPtr = S_mapMat->values;
        double *s_mapPtr = (double *) S_map.data;

        for (int i = 0; i < S_map.rows * S_map.cols; i++)
            *s_mapPtr++ = (double) std::abs(*s_matrixPtr++);

        GPLib::writeCvMatToFile<double>(S_map, "S_map.yml", false, false);

        if (m_metricPar.do_spatial_pooling) {
            double A = cv::sum(S_map)[0];
            double min, max;
            cv::minMaxLoc(S_map, &min, &max);

        S_map = A / max * S_map;
        }

        s_mapPtr = (double *) S_map.data;
        double *p_mapPtr = (double *) P_map.data;

        for (int i = 0; i < S_map.rows * S_map.cols; i++)
            *p_mapPtr++ = 1.0 - exp(log(0.5)*(*s_mapPtr++));
        
        double min;
        cv::minMaxLoc(P_map, &min, &P_det);

        S_map.copyTo(C_map);
        cv::minMaxLoc(C_map, &min, &C_max);

        Q_MOS = 100.0 / ( 1.0 + exp(m_metricPar.quality_logistic_q1 * (Q+m_metricPar.quality_logistic_q2))); 

        std::cout << "-----------------------------------------\n";
        std::cout << "C_max: " << C_max << std::endl;
        std::cout << "Q:     " << Q << std::endl;
        std::cout << "Q_MOS: " << Q_MOS << std::endl;
        std::cout << "-----------------------------------------\n";
    }


    void HDRVDP::loadImages(std::string referenceName, std::string testName, COLOR_ENC color_encoding) throw (std::runtime_error) {
        if ((color_encoding == dip::SRGB_DISPLAY) || (color_encoding == dip::RGB_BT_709)) {
            m_refMat = cv::imread(referenceName, CV_LOAD_IMAGE_COLOR);
            m_testMat = cv::imread(testName, CV_LOAD_IMAGE_COLOR);
        } else {
            m_refMat = cv::imread(referenceName, CV_LOAD_IMAGE_GRAYSCALE);
            m_testMat = cv::imread(testName, CV_LOAD_IMAGE_GRAYSCALE);
        }
    }


    void HDRVDP::setSources() throw (std::runtime_error, cl::Error) {
        if (m_refMat.rows == 0 || m_refMat.cols == 0)
            throw std::runtime_error("ERROR: setSources() - (m_refMat.rows == 0 || m_refMat.cols == 0)");
        if (m_refMat.rows != m_testMat.rows || m_refMat.cols != m_testMat.cols)
            throw std::runtime_error("ERROR: setSources() - (m_refMat.rows != m_testMat.rows || m_refMat.cols != m_testMat.cols)");

        if ((m_color_encoding == dip::SRGB_DISPLAY) || (m_color_encoding == dip::RGB_BT_709)) {
            cv::cvtColor(m_refMat, m_refMat, CV_BGR2RGB);
            cv::cvtColor(m_testMat, m_testMat, CV_BGR2RGB);

            m_refMat.convertTo(m_refMat, CV_64FC3);
            m_testMat.convertTo(m_testMat, CV_64FC3);

            //m_refMat.convertTo(m_refMat, CV_64FC3, 1 / 255.0);
            //m_testMat.convertTo(m_testMat, CV_64FC3, 1 / 255.0);
            if (m_color_encoding == dip::SRGB_DISPLAY)
                    display_mode_sRGB();
        }
        else {
            if (m_testMat.channels() > 1 || m_refMat.channels() > 1)
                throw std::runtime_error("ERROR: setSources() - (LUMINANCE || LUMA) && (m_testMat.channels() > 1 || m_refMat.channels() > 1)");

            m_refMat.convertTo(m_refMat, CV_64FC1, 1/255.0);
            m_testMat.convertTo(m_testMat, CV_64FC1, 1/255.0);

            if (m_color_encoding == dip::LUMA_DISPLAY)
                display_mode_Luma();
        }

        m_testMat.copyTo(m_testBackup);
   }

    void HDRVDP::display_mode_Luma() {
        double gamma = 2.2;
        double peak = 99;
        double black_level = 1.0;

        double *testPtr = (double *) (m_testMat.data);
        double *refPtr = (double *) (m_refMat.data);

        size_t numOfPixels = m_testMat.rows * m_testMat.cols;

        for(size_t i = 0; i < numOfPixels; i++) {
            *testPtr++ = peak * std::pow(*testPtr, gamma) + black_level;
            *refPtr++ = peak * std::pow(*refPtr, gamma) + black_level;
        }
    }

    void HDRVDP::display_mode_sRGB() throw (cl::Error) {
        int deviceDataSize;

        if (clObj->devSupportDouble())
            deviceDataSize = m_refMat.rows * m_refMat.cols * m_refMat.channels() * sizeof(double);
        else {
            deviceDataSize = m_refMat.rows * m_refMat.cols * m_refMat.channels() * sizeof(float);
            m_refMat.convertTo(m_refMat, CV_32FC3);
            m_testMat.convertTo(m_testMat, CV_32FC3);
        }

        cl::Buffer inBuffer1(*clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
        cl::Buffer inBuffer2(*clObj->getContext(), CL_MEM_READ_WRITE, deviceDataSize);
        
        cl::Kernel kernel(*clObj->getProgram(), "display_sRGB");

        (*clObj->getQueue()).enqueueWriteBuffer(inBuffer1, CL_TRUE, 0, deviceDataSize, m_refMat.data);
        (*clObj->getQueue()).enqueueWriteBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_testMat.data);
        
        kernel.setArg(0, inBuffer1);
        kernel.setArg(1, inBuffer2);

        (*clObj->getQueue()).enqueueNDRangeKernel(kernel, cl::NullRange, \
                                     cl::NDRange(m_refMat.rows * m_refMat.cols * m_refMat.channels()));

        (*clObj->getQueue()).finish();

        (*clObj->getQueue()).enqueueReadBuffer(inBuffer1, CL_TRUE, 0, deviceDataSize, m_refMat.data);
        (*clObj->getQueue()).enqueueReadBuffer(inBuffer2, CL_TRUE, 0, deviceDataSize, m_testMat.data);

        m_refMat.convertTo(m_refMat, CV_64FC3);
        m_testMat.convertTo(m_testMat, CV_64FC3);
    }


    void HDRVDP::load_spectral_resp() throw (std::runtime_error) {
        std::string fileName;
        unsigned int rows, cols;

        switch (m_color_encoding) {
            case dip::SRGB_DISPLAY:
            case dip::RGB_BT_709:
                cols = 3;
                fileName = "IMG_E_ccfl-lcd.csv";
                break;
            case dip::LUMINANCE:
                cols = 1;
                fileName = "IMG_E_d65.csv";
                break;
            default:
                break;
        }

        std::ifstream file(fileName);
        rows = static_cast<unsigned int>(std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n'));
        
        if (rows == 0) {
            throw std::runtime_error("ERROR: IMG_E_ccfl-lcd.csv or IMG_E_d65.csv not found!");
        }

        file.clear();
        file.seekg(0, std::ios::beg);

        m_metricPar.spectral_emission = cv::Mat(rows, cols, CV_64FC1);

        double *dataPtr = (double*) m_metricPar.spectral_emission.data;

        int i = -1;
        for(dip::CSVIterator loop(file);loop != dip::CSVIterator();++loop) {
            dataPtr[++i] = atof((*loop)[0].c_str());
            if ((m_color_encoding == dip::SRGB_DISPLAY) || (m_color_encoding == dip::RGB_BT_709)) {
                dataPtr[++i] = atof((*loop)[1].c_str());
                dataPtr[++i] = atof((*loop)[2].c_str());
            }
        }
       
        return;
    }

 
    cv::Mat HDRVDP::mutual_masking(cv::Mat &testBand, cv::Mat &refBand) {
        cv::Mat result(testBand.rows, testBand.cols, testBand.type());

        double *resPtr = (double*) result.data;
        double *testPtr = (double*) testBand.data;
        double *refPtr = (double*) refBand.data;

        for (int i = 0; i < result.rows * result.cols; i++)
            *resPtr++ = std::min(std::abs(*testPtr++), std::abs(*refPtr++));

        cv::Mat F = cv::Mat::ones(3, 3, CV_64FC1) / 9.0;

        cv::filter2D(result, result, result.type() , F, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT);

        return result;
    }


    cv::Mat HDRVDP::get_band(PYRAMID pyr, int band, int orienation) {
        MATRIX tmpMatrix;
        float minOne = 1.0f;

        if (band == 0)
            tmpMatrix = pyr->hiband;
        else if (band == pyr->num_levels + 1)
            tmpMatrix = pyr->lowband;
        else {
            tmpMatrix = GetSubbandImage(pyr, band - 1, orienation);
            //minOne = -1.0f; //!!!!!!!!
        }

        cv::Mat result(tmpMatrix->rows, tmpMatrix->columns, CV_32FC1);

        float *resPtr = (float *) result.data;
        float *tmpPtr = tmpMatrix->values;

        for (int i = 0; i < tmpMatrix->rows * tmpMatrix->columns; i++) {
            *resPtr++ = minOne * (*tmpPtr++);
        }

        //cv::Mat tmpMat(tmpMatrix->rows, tmpMatrix->columns, CV_32FC1, tmpMatrix->values);
        //result = tmpMat.clone();
        result.convertTo(result, CV_64FC1);

        return result;
    }


    void HDRVDP::set_band(PYRAMID pyr, int band, int orienation, const cv::Mat &newBand) {
        MATRIX tmpMatrix;
        cv::Mat tmpBand;

        if (band == 0)
            tmpMatrix = pyr->hiband;
        else if (band == pyr->num_levels + 1)
            tmpMatrix = pyr->lowband;
        else
            tmpMatrix = GetSubbandImage(pyr, band - 1, orienation);

        newBand.convertTo(tmpBand, CV_32FC1);

        float *newBandPtr = (float *) tmpBand.data;
        float *tmpPtr = tmpMatrix->values;

        for (int i = 0; i < tmpMatrix->rows * tmpMatrix->columns; i++)
            *tmpPtr++ = *newBandPtr++;
    }


    void HDRVDP::copyPyramid(const PYRAMID src, PYRAMID dst) {
        dst->num_levels = src->num_levels;
        dst->hiband = CopyMatrix(src->hiband);
        dst->lowband = CopyMatrix(src->lowband);
        dst->levels = new PBAND[src->num_levels];

        for (int i = 0; i < src->num_levels; i++) {
            dst->levels[0]->subband = new MATRIX;

            (dst->levels[0]->subband) = new MATRIX(*(src->levels[i]->subband));
            //*(dst->levels[i]->subband) = CopyMatrix(*(src->levels[0]->subband));
            //memcpy(dst->levels[i]->subband, tmp, tmp->rows * tmp->columns * sizeof(_matrix));
        }

    }
}
