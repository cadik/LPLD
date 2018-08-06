#include "bow_params.hpp"

namespace dip {
    void BOW_PARAMS::loadSources(const char *dicitionary) throw (std::runtime_error) {
        std::ifstream file(dicitionary);
        m_dictionaryRows = static_cast<size_t>(std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n'));
        
        if (m_dictionaryRows == 0) {
            throw std::runtime_error("ERROR: Dictionary not found!");
        }

        file.clear();
        file.seekg(0, std::ios::beg);

        unsigned int y = 0;
        for(dip::CSVIterator loop(file);loop != dip::CSVIterator();++loop) {
            for(unsigned int x = 0; x < PATCH_SIZE; x++) {
                m_dictionary.at<double>(y, x) = atof((*loop)[x].c_str());
            }
            y++;
        }
    }

    void BOW_PARAMS::initDCT_ZIGZAG() {
        for(unsigned int zz = 0; zz < PATCH_SIZE; zz++) {
            for(unsigned int rr = 0; rr <= zz; rr++) {
                int x = zz-rr;
                int y = zz-x;
                m_dctZigZagOrder.push_back(std::pair<int, int>(x, y));
            }
        }
        for(unsigned int zz = PATCH_SIZE; zz < (2*PATCH_SIZE); zz++) {
            int offset = zz - PATCH_SIZE;
            for(unsigned int rr = 1 + offset; rr < zz-offset; rr++) {
                int x = zz-rr;
                int y = zz-x;
                m_dctZigZagOrder.push_back(std::pair<int, int>(x, y));
            }
        }
    }

}