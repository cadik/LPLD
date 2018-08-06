#ifndef DIP_DECISION_FOREST_HPP
#define DIP_DECISION_FOREST_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#define NOMINMAX

#endif

#include "binaryTree.hpp"
#include "algStrategy.hpp"

#include "stdafx.h"
#include <fstream>
#include <dirent.h>
#include <regex>
#include <sstream>

#include <opencv2/core/core.hpp>

namespace dip {
    class DecisionForest {
    public:
        DecisionForest(const char* forestPath) throw (std::runtime_error);
        ~DecisionForest() {
            for(std::vector<BST *>::iterator it = m_forest.begin(); it != m_forest.end(); ++it)
                delete (*it);
        }

        cv::Mat predict(const std::vector<cv::Mat> &methodsResults) throw (std::runtime_error);

    private:
        std::string m_decisionForestPath;
        std::vector<std::string> m_tree_files;
        std::vector<BST *> m_forest;

        void loadTrees() throw (std::runtime_error);
    };
}

#endif
