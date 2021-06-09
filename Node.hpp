#include <stdlib.h> 
#include "Model.hpp"

class Node {
    public:
        int ID;                 // ID of node, should correspond to index in list of nodes
        int y;                  // predicted label of a node

        int dim_features;       // input/feature dimension
        int dim_hidden;         // hidden dimension
        int num_classes;        // number of classes
        int degree;             // degree of node 

        float* x;               // attribute of node 
        float* tmp_hidden;      // holds copy of hidden representation (for aggregation)
        float* hidden;          // holds hidden representation (i.e. after transformation and aggregation)
        float* tmp_logits;      // holds copy of logits (for aggregation)
        float* logits;          // holds logits (i.e. after transformation and aggregation)

        std::list<int> neighbors;

        Node(int id, Model &model, bool allocate_memory);
        void free_node(void);
        void load_input(std::string dir);
        int get_prediction(void);
};