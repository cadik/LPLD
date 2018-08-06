#include <iostream>

#include "GetoptLib.hpp"
#include "CLSharedLib.hpp"
#include "GeneralPurposeLib.hpp"
#include "diff.hpp"
#include "grad_dist.hpp"
#include "hog9.hpp"
#include "mask_entropy_I.hpp"
#include "mask_entropy_multi.hpp"
#include "ssim_struct.hpp"
#include "ssim_lum.hpp"
#include "harris.hpp"
#include "spyrDist.hpp"
#include "hdrvdp_band.hpp"
#include "bow.hpp"
#include "decision_forest.hpp"
#include "algStrategy.hpp"
#include <string>
#include <vector>

void help();
void winPresEnter();

int main(int argc, char **argv)
{
    static struct option long_options[] = {
        {"referece",  ARG_REQ, 0, 'r'},
        {"test",     ARG_REQ, 0, 't'},
        {"help",  ARG_NONE,  0, 'h'},
        { ARG_NULL , ARG_NULL , ARG_NULL , ARG_NULL }
    };

    int platformNum = 0, deviceNum = 0;
    int c;
    std::string reference_img;
    std::string test_img;

    while (1) {
        int option_index = 0;
        c = CmdLn::getopt_long(argc, argv, "hr:t:", long_options, &option_index);

        // Check for end of operation or error
        if (c == -1)
            break;

        // Handle options
        switch (c) {
        case ('r'):
            reference_img = optarg;
            break;
        case ('t'):
            test_img = optarg;
            break;
        case ('h'):
            help();
            return 0;

        case '?':
            help();
            return 0;

        default:
            help();
            return 0;
        }
    }

    if (optind < argc) {
        help();
        return 0;
    }

    if (reference_img.size() == 0 || test_img.size() == 0) {
        std::cout << "INFO: Plese specify all input images.\n";
        std::cout << "INFO: --help or -h for help.\n";

        winPresEnter();
        return 0;
    }

    std::cout << "Running diff with options: \n";
    std::cout << "Reference image: " << reference_img << "\n" << \
        "Test image: " << test_img << "\n";

    try {
        using namespace dip;

        std::vector<Context> methods;

        cv::Mat ref_Mat = cv::imread(reference_img, CV_LOAD_IMAGE_COLOR);
        cv::Mat test_Mat = cv::imread(test_img, CV_LOAD_IMAGE_COLOR);

        MASK_ENT_I maskE_I_Method(test_Mat, ref_Mat, 0, CL_DEVICE_TYPE_CPU, 0);
        SPYR_DIST spyrDist_Method(test_Mat, ref_Mat);
        MASK_ENT_MULTI maskE_multiple_Method(test_Mat, ref_Mat, 0, 0, CL_DEVICE_TYPE_CPU, 0);
        HARRIS harris_Method(test_Mat, ref_Mat, 5);
        SSIM_LUM ssim_lum_Method(test_Mat, ref_Mat, 0, CL_DEVICE_TYPE_CPU, 0);
        DIFF diff_Method(test_Mat, ref_Mat, 0, CL_DEVICE_TYPE_CPU, 0);
        Grad_Dist gradDist_Method(test_Mat, ref_Mat, 0, CL_DEVICE_TYPE_CPU, 0);
        SSIM_STRUCT ssim_struct_Method(test_Mat, ref_Mat, 0, CL_DEVICE_TYPE_CPU, 0);
        HOG9 hog9_Method(test_Mat, ref_Mat);
        HDRVDP_BAND hdrvdpBand_Method(test_Mat, ref_Mat, 4, 0, CL_DEVICE_TYPE_CPU, 0);
        
        BOW_PARAMS params("ARTIFACT_DICTIONARY.csv", 16);
        BOW bow_Method(test_Mat, ref_Mat, params, 4, 0, CL_DEVICE_TYPE_CPU, 0);
		///*
        //methods.push_back(Context(maskE_I_Method)); //3.7 ok
        //methods.push_back(Context(spyrDist_Method));//4.8 nic
        //methods.push_back(Context(maskE_multiple_Method)); //3.8 divne
        //methods.push_back(Context(harris_Method));  //3 nic
        //methods.push_back(Context(ssim_lum_Method));	//1.4 huh
        //methods.push_back(Context(ssim_struct_Method)); //2.7
        //methods.push_back(Context(diff_Method));	//nothing
        //methods.push_back(Context(gradDist_Method));//5.2
        //methods.push_back(Context(hog9_Method)); //0,3
        methods.push_back(Context(hdrvdpBand_Method));//*/
        methods.push_back(Context(bow_Method)); // too much


#ifdef _MSC_VER
        const char *decisionForestPath = "decision_forest\\";
#else
        const char *decisionForestPath = "decision_forest/";
#endif

       
        
        std::vector<cv::Mat> methodsResults;
		for (int i = 0; i < methods.size(); i++) {
			methods[i].compute();
 			methodsResults.push_back(methods.at(i).getResult());
			auto img = methods.at(i).getResult();
			cv::imshow("result", methods.at(i).getResult());

			
			//GPLib::writeCvMatToFile<double>(img, "matrixes/matrix.yml", true);
			cv::waitKey();
		}
      
		
		DecisionForest forest(decisionForestPath);

        cv::Mat result = forest.predict(methodsResults);

        cv::imshow("result", result);
        cv::waitKey();

        //winPresEnter();

    } catch(std::runtime_error error) {
        std::cerr << error.what() << std::endl;
        winPresEnter();
        return -1;
    } catch(std::out_of_range error) {
        std::cerr << error.what() << std::endl;
        winPresEnter();
        return -1;
    } catch(...) {
        std::cerr << "ERROR: Catch unhandled exception! (main.cpp)" << std::endl;
        winPresEnter();
        return -1;
    }

    return 0;
}


void help() {
    std::cout << "Usage: ./lpld [options] \n  where options are:\n";
    std::cout << "where options are:\n";
    std::cout << "--help or -h\n";
    std::cout << "--referece <image1> or -r <image1>               -> Path to image1 *Required\n";
    std::cout << "--test <image2> or -t <image2>                   -> Path to image2 *Required\n";

    winPresEnter();

    return;
}


void winPresEnter() {
#ifdef _MSC_VER
    std::cout << "Press Enter...\n";
    std::cin.ignore();
#endif
    return;
}

