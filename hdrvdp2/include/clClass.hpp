#ifndef DIP_CLCLASS_HPP
#define DIP_CLCLASS_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <CL/cl.hpp>

#include "CLSharedLib.hpp"

namespace dip {
    class clClass {
    public:
        clClass(int platformNum = 0, cl_device_type deviceType = CL_DEVICE_TYPE_ALL, int deviceNum = 0)\
                throw(std::runtime_error, cl::Error);
        
        const cl::Context *getContext() {return &m_context;}
        const cl::CommandQueue *getQueue() {return &m_queue;}
        const cl::Program *getProgram() {return &m_program;}
        bool devSupportDouble() { return m_dev_doubleSupport; }

    private:
        cl::Platform m_platform;
        cl::Context m_context;
        cl::Device m_device;
        cl::CommandQueue m_queue;
        cl::Program m_program;

        bool m_dev_doubleSupport;
    };
}

#endif