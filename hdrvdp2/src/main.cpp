#include <iostream>
#include "GetoptLib.hpp"
#include "hdrvdpHelper.hpp"
#include "hdrvdp.hpp"
#include "hdrvdpVisualize.hpp"


void help();
void winPresEnter();

int main(int argc, char** argv)
{
    static struct option long_options[] = {
            {"referece",  ARG_REQ, 0, 'r'},
            {"test",     ARG_REQ, 0, 't'},
            {"cl-platform",  ARG_REQ,  0, 'p'},
            {"cl-device",  ARG_REQ,  0, 'd'},
            {"cl-info",  ARG_NONE,  0, 'i'},
            {"help",  ARG_NONE,  0, 'h'},
            { ARG_NULL , ARG_NULL , ARG_NULL , ARG_NULL }
    };

    int platformNum = 0, deviceNum = 0;
    int c;
    std::string reference_img;
    std::string test_img;

    while (1) {		
        int option_index = 0;
        c = CmdLn::getopt_long(argc, argv, "hir:t:p:d:", long_options, &option_index);

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
            case ('p'):
                platformNum = atoi(optarg);
                break;
            case ('d'):
                deviceNum = atoi(optarg);
                break;
            case ('i'):
                dip::CLInfo::printInfo();
                winPresEnter();
                return 0;
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

    std::cout << "Running HDRVDP with options: \n";
    std::cout << "Reference image: " << reference_img << "\n" << \
                 "Test image: " << test_img << "\n";

    try {
        dip::HDRVDP hdrvpObj(platformNum, CL_DEVICE_TYPE_ALL, deviceNum);

        //double ppd = dip::HDRVDP_helper::pix_per_deg(21, cv::Size(1,1), 1);
        double ppd = 60.0;
        hdrvpObj.compute(reference_img, test_img, dip::COLOR_ENC::RGB_BT_709, ppd);

        cv::Mat P_map = hdrvpObj.getP_Map();
        cv::Mat testImg = hdrvpObj.getTestImage();

        dip::HDRVD_Visualize vizObj(P_map, testImg);
        
        cv::imshow("hdrvdp_visualize", vizObj.getMap());
        cv::waitKey();

    } catch (cl::Error error) {
        std::cerr << dip::CLInfo::get_error_string(error.err()) << std::endl;
        winPresEnter();
        return -1;
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
    std::cout << "Usage: ./hdrvdp [options] \n  where options are:\n";
    std::cout << "where options are:\n";
    std::cout << "--help or -h\n";
    std::cout << "--referece <image1> or -r <image1>               -> Path to image1 *Required\n";
    std::cout << "--test <image2> or -t <image2>                   -> Path to image2 *Required\n";
    std::cout << "--cl-platform <platformNum> or -p <platformNum>  -> Set OpenCL platform. *Def. 0\n";
    std::cout << "--cl-device <devNum> or -d <devNum>              -> Set OpenCL device. *Def. 0\n";
    std::cout << "--cl-info or -i                                  -> Print info about available platforms and devices.\n";

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
