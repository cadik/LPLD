#include "binaryTree.hpp"

namespace dip {
    BST::node* BST::CreateNode(int nodeID, double key, int methodNum, \
                               node* leftChild, node* rightChild) {
        node *n = new node;
        
        n->nodeID = nodeID;
        n->key = key;
        n->methodNum = methodNum;

        if ((leftChild == nullptr) && (rightChild == nullptr)) {
            n->leftPtr = nullptr;
            n->rightPtr = nullptr;
        } else {
            n->leftPtr = leftChild;
            n->rightPtr = rightChild;
        }

        return n;
    }


    void BST::AddLeaf(int nodeID, double key) {
        node *newLeaf = CreateNode(nodeID, key);
        leafNodesReservoir.push_back(newLeaf);
    }


    void BST::AddNode(int nodeID, double key, int methodNum, int leftChildID, int rightChildID) {
        node *tmpLeftChild, *tmpRightChild;
        int leftChildPos = -1, rightChildPos = -1;

        for(std::size_t i = 0; i != leafNodesReservoir.size(); i++) {
            if (leafNodesReservoir[i]->nodeID == rightChildID) {
                tmpRightChild = leafNodesReservoir[i];
                rightChildPos = i;
            }

            if (leafNodesReservoir[i]->nodeID == leftChildID) {
                tmpLeftChild = leafNodesReservoir[i];
                leftChildPos = i;
            }

            if ((leftChildPos != -1) && (rightChildPos != -1))
                break;
        }

        node *tmp_root = CreateNode(nodeID, key, methodNum, tmpLeftChild, tmpRightChild);

        leafNodesReservoir.erase(leafNodesReservoir.begin() + rightChildPos);
        leafNodesReservoir.erase(leafNodesReservoir.begin() + leftChildPos - 1);

        leafNodesReservoir.push_back(tmp_root);

        if (leafNodesReservoir.size() == 1)
            root = tmp_root;
    }


    double BST::predict(std::vector<double> &methodsVals) {
        node *tmpRoot = root;
        if (tmpRoot == nullptr)
            return 0.0;

        while(tmpRoot->leftPtr != nullptr) {
            double methodVal = methodsVals[(tmpRoot->methodNum)-1]; //puvodni bey minus 1
            tmpRoot = (methodVal < tmpRoot->key) ? tmpRoot->leftPtr: tmpRoot->rightPtr;
        }

        return tmpRoot->key;
    }

    BST::~BST() {
        RemoveSubtree(root);
    }


    void BST::RemoveSubtree(node* nodePtr) {
        if (nodePtr == nullptr)
            return;

        if (nodePtr->leftPtr != nullptr)
            RemoveSubtree(nodePtr->leftPtr);
        if (nodePtr->rightPtr != nullptr)
            RemoveSubtree(nodePtr->rightPtr);

        delete nodePtr;
    }
}