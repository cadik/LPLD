#include "decision_forest.hpp"

namespace dip {
    DecisionForest::DecisionForest(const char* forestPath) throw (std::runtime_error) {
        m_decisionForestPath = forestPath;

        DIR *dir;
        struct dirent *ent;

#ifdef _MSC_VER
        const char *regexExpr = "(tree\_[[:digit:]]+\.csv)";
#else
        const char *regexExpr = "(tree\\_[[:digit:]]+\\.csv)";
#endif

        if ((dir = opendir(forestPath)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                if (std::regex_match(ent->d_name, std::regex(regexExpr) ))
                    m_tree_files.push_back(std::string(ent->d_name));
            }
            closedir(dir);
        } else
            throw std::runtime_error("ERROR: Could not open directory with decision forest!");

        loadTrees();
    }


    void DecisionForest::loadTrees() throw (std::runtime_error) {
        std::stringstream ss;
        std::vector<double> valuesVec;
        std::string item;
        int nodesNum = 0;

        for(std::vector<std::string>::const_iterator it = m_tree_files.begin(); it != m_tree_files.end(); ++it) {
            std::fstream file(m_decisionForestPath + (*it));
            std::string tmpStr;

            BST *tree = new BST();

            while (std::getline(file, tmpStr))
            {
                ss << tmpStr;
                while (std::getline(ss, item, ';')) {
                    valuesVec.push_back(std::stod(item));
                }

                int nodeID = static_cast<int>(valuesVec[0]);
                double nodeValue = valuesVec[4];

                if (valuesVec[1] == 0.0) {
                    nodesNum++;
                    tree->AddLeaf(nodeID, nodeValue);
                }
                else {
                    nodesNum++;
                    int methodNum = static_cast<int>(valuesVec[5]);
					//PATCHWORK JUST TESTING  --- REMOVE
					if (methodNum == 12) {
						methodNum = 10;
					}

					//endPATCHOWRK
                    int leftChildID = static_cast<int>(valuesVec[2]);
                    int rightChildID = static_cast<int>(valuesVec[3]);
                    tree->AddNode(nodeID, nodeValue, methodNum, leftChildID, rightChildID);
                }

                ss.clear();
                valuesVec.clear();
            }

            std::cout << (*it) << " -> nodes: " << nodesNum << std::endl;
            nodesNum = 0;

            m_forest.push_back(tree);
            file.close();
        }
    }


	cv::Mat DecisionForest::predict(const std::vector<cv::Mat> &methodsResults) throw (std::runtime_error) {
#if 1
		std::vector<double> methodsResPtrs(0);
		cv::Mat result(methodsResults.at(0).rows, methodsResults.at(0).cols, CV_64FC1);
		for (int i = 0; i < methodsResPtrs.size(); i++) {
			
		}
        double *resultPtr = (double *) result.data;

        for(int pixPos = 0; pixPos < result.rows * result.cols; pixPos++) {
            for(unsigned int i = 0; i < methodsResults.size(); i++)
                methodsResPtrs.push_back(*(((double *) methodsResults[i].data) + pixPos));
            
            double trees_predictions = 0.0;
            for(std::vector<BST *>::iterator it = m_forest.begin(); it != m_forest.end(); ++it) {
                 trees_predictions += (*it)->predict(methodsResPtrs);
            }

            trees_predictions /= m_forest.size();

            *resultPtr++ = trees_predictions;

            methodsResPtrs.clear();
        }

        return result;
#else
        return methodsResults.at(0);
#endif
    }
}
