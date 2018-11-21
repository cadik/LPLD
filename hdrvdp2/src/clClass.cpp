#include "clClass.hpp"

namespace dip {
    // TODO! : srcFile for 32/64 floating point
    clClass::clClass(int platformNum, cl_device_type deviceType, int deviceNum) \
                     throw (std::runtime_error, cl::Error) {

        std::vector<cl::Platform> platformsList;
        std::vector<cl::Device> devicesList;

        try {
            cl::Platform::get(&platformsList);

            try {
                m_platform = platformsList.at(platformNum);
            }
            catch (const std::out_of_range& oor) {
                std::cerr << "Out of Range error (std::vector<cl::Platform>)!\n" << \
                             "Please choose correct cl::Platform number (--cl-platform param)!\n";
                throw oor;
            }


            dip::CLInfo::printQuickPlatformInfo(m_platform, platformNum);

            cl_context_properties cprops[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) m_platform(), 0};
            m_context = cl::Context(deviceType, cprops);

            std::vector<cl::Device> tmp_devicesList = m_context.getInfo<CL_CONTEXT_DEVICES>();
            
            try {
                m_device = tmp_devicesList.at(deviceNum);
            }
            catch (const std::out_of_range& oor) {
                std::cerr << "Out of Range error (std::vector<cl::Device>)!\n" << \
                             "Please choose correct cl::Device number (--cl-device param)!\n";
                throw oor;
            }

            

            devicesList.push_back(m_device);

            m_dev_doubleSupport = dip::CLInfo::hasDoublePrecision(m_device);
            
            // !!!!!!!!!!!!!
            std::string srcFile;
            if (m_dev_doubleSupport)
                srcFile = "hdrvdp_64fp.cl";
            else
                srcFile = "hdrvdp_32fp.cl";

            dip::CLInfo::printQuickDeviceInfo(m_device, deviceNum);

            m_queue = cl::CommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE);

#ifdef _MSC_VER 
            std::ifstream sourceFile("src\\" + srcFile);
#else
            std::ifstream sourceFile("src/" + srcFile);
#endif

            std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), \
                                    (std::istreambuf_iterator<char>()));

            cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

            m_program = cl::Program(m_context, source);

            m_program.build(devicesList);
        } catch (cl::Error error) {
            if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:" << std::endl << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesList[0]) << std::endl;
            }
            std::cerr << "ERROR: InitCL\n";
            throw;
        } catch (...) {
            throw;
        }
    }
}
