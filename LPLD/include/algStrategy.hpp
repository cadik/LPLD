#ifndef DIP_ALG_STRATEGY_HPP
#define DIP_ALG_STRATEGY_HPP

#ifdef _MSC_VER 
#define NOMINMAX
#endif

#include "stdafx.h"

#include "opencv2/core/core.hpp"

namespace dip {
    class methodStrategy {
    public:
        virtual void compute() = 0;
        virtual const cv::Mat getResult() = 0;
    };


    class Context {
        methodStrategy &m_strategy;

    public:
        Context(methodStrategy &strategy) : m_strategy(strategy) {}
        void compute() {
            m_strategy.compute();
        }

        const cv::Mat getResult() {
            return m_strategy.getResult();
        }
    };
}

#endif
