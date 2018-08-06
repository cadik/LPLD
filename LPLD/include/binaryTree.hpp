#ifndef DIP_BIN_TREE_HPP
#define DIP_BIN_TREE_HPP

#ifdef _MSC_VER 
    #pragma warning(disable : 4290)
#define NOMINMAX
#endif

#include <vector>

namespace dip {
    class BST {
    private:
        struct node
        {
            int nodeID;
            double key;
            int methodNum;
            node* leftPtr;
            node* rightPtr;
        };

        node *root;
        std::vector<node *> leafNodesReservoir;

        node* CreateNode(int nodeID, double key, int methodNum = 0, node* leftChild = nullptr, node* rightChild = nullptr);
        void RemoveSubtree(node* nodePtr);

    public:
        BST() {root = nullptr;}
        ~BST();
        void AddLeaf(int nodeID, double key);
        void AddNode(int nodeID, double key, int methodNum, int leftChildID, int rightChildID);
        double predict(std::vector<double> &methodsVals);
    };
}

#endif
