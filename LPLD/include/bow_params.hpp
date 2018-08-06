#ifndef DIP_BOW_PARAMS_HPP
#define DIP_BOW_PARAMS_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include "stdafx.h"

#include "csvIterator.hpp"

namespace dip {
    class BOW_PARAMS {
    public:
        BOW_PARAMS(const char *dicitionary, size_t patchSize) throw (std::runtime_error)  {
            if (patchSize < 1)
                throw std::runtime_error("ERROR: patchSize < 1!");
            if (dicitionary && dicitionary[0] == '\0')
                throw std::runtime_error("ERROR: dictionary string is empty!");

            CANONICAL_IMAGE_RESOLUTION = 640;
            PATCH_SIZE = patchSize;
            DICTIONARY_SIZE = 32;
            LUMINANCE_INVARIANT_PATCHES = true;
            CONTRAST_INVARIANT_PATCHES = true;
            m_maxPatchSize = 48;
            m_dictionary = cv::Mat(DICTIONARY_SIZE, patchSize, CV_64FC1);
            m_stride = 2;

            loadSources(dicitionary);
            initDCT_ZIGZAG();
        }

        ~BOW_PARAMS() {
        }

        cv::Mat getDictionary() const {return m_dictionary;}
        size_t getCanonicalImgRes() const {return CANONICAL_IMAGE_RESOLUTION;}
        size_t getPatchSize() const {return PATCH_SIZE;}
        size_t getDictionarySize() const {return DICTIONARY_SIZE;}
        size_t getStride() const {return m_stride;}
        bool lumInvariantPatch() const {return LUMINANCE_INVARIANT_PATCHES;}
        bool contrastInvariantPatch() const {return CONTRAST_INVARIANT_PATCHES;}
        std::vector<std::pair<int,int>> getDCTZigZagOrder() const { return m_dctZigZagOrder;}


    private:
        BOW_PARAMS();
        void loadSources(const char *dicitionary) throw (std::runtime_error);
        void initDCT_ZIGZAG();

        cv::Mat m_dictionary;
        size_t m_dictionaryRows;

        std::vector<std::pair<int, int>> m_dctZigZagOrder;

        size_t CANONICAL_IMAGE_RESOLUTION; 
        size_t PATCH_SIZE;
        size_t DICTIONARY_SIZE;

        bool LUMINANCE_INVARIANT_PATCHES;
        bool CONTRAST_INVARIANT_PATCHES;

        size_t m_maxPatchSize;

        size_t m_stride;
    };
}

#endif
