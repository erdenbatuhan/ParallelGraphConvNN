#include "Model.hpp"
#include "Node.hpp"


#define DEBUG 0


/***************************************************************************************/
void first_layer_transform(Node** nodes, int num_nodes, Model &model){
    Node* node;
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        for(int c_out = 0; c_out < node->dim_hidden; ++c_out){
            for(int c_in = 0; c_in < node->dim_features; ++c_in){
                node->tmp_hidden[c_out] += node->x[c_in] * model.weight_1[c_in * node->dim_hidden + c_out];
            }
        }   
    }
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_aggregate(Node** nodes, int num_nodes, Model &model){
    // aggregate
    float* message;
    float norm;
    Node* node;

    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        // aggregate from each neighbor
        for(int neighbor : node->neighbors){
            message = nodes[neighbor]->tmp_hidden;
            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message
            for(int c = 0; c < node->dim_hidden; ++c){
                node->hidden[c] += message[c] / norm;
            }
        }
        // add bias
        for(int c = 0; c < node->dim_hidden; ++c){
            node->hidden[c] += model.bias_1[c];
        }
        // apply relu
        for(int c = 0; c < node->dim_hidden; ++c){
            node->hidden[c] = node->hidden[c] < 0.0 ? 0.0 : node->hidden[c];
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
// computation in second layer
void second_layer_transform(Node** nodes, int num_nodes, Model &model){
    // transform
    Node* node;
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        for(int c_out = 0; c_out < node->num_classes; ++c_out){
            for(int c_in = 0; c_in < node->dim_hidden; ++c_in){
                node->tmp_logits[c_out] += node->hidden[c_in] * model.weight_2[c_in * node->num_classes + c_out];
            }
        }   
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_aggregate(Node** nodes, int num_nodes, Model &model){
    // aggregate
    Node* node;
    float* message;
    float norm;
    // for each node
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        // aggregate from each neighbor
        for(int neighbor : node->neighbors){
            message = nodes[neighbor]->tmp_logits;
            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message
            for(int c = 0; c < node->num_classes; ++c){
                node->logits[c] += message[c] / norm;
            }
        }

        // add bias
        for(int c = 0; c < node->num_classes; ++c){
            node->logits[c] += model.bias_2[c];
        }
    }        
}
/***************************************************************************************/


/***************************************************************************************/
void create_graph(Node** nodes, Model &model){
    // set neighbor relations
    int source, target;
    for(int e = 0; e < model.num_edges; ++e){
        source = model.edges[e];
        target = model.edges[model.num_edges + e];
        // self-loops twice in edges, so ignore for now
        // and add later
        if (source != target){
            nodes[source]->neighbors.push_back(target);
        }
    }

    // add self-loops
    for(int n = 0; n < model.num_nodes; ++n){
        Node *node = nodes[n];
        node->neighbors.push_back(node->ID);
        node->degree = node->neighbors.size();
    }
}
/***************************************************************************************/


/***************************************************************************************/
int main(int argc, char** argv) {
    int seed = -1;
    int init_no = -1;
    std::string dataset("");

    // specify problem
    #if DEBUG
        // for measuring your local runtime
        auto tick = std::chrono::high_resolution_clock::now();
        Model::specify_problem(argc, argv, dataset, &init_no, &seed);
    #else
        Model::specify_problem(dataset, &init_no, &seed);
    #endif

    // load model specifications and model weights
    Model model(dataset, init_no, seed);
    model.load_model();

    // create graph (i.e. load data into each node and load edge structure)
    Node** nodes = (Node**)malloc(model.num_nodes * sizeof(Node*));
    if(nodes == nullptr){
        exit(1);
    }
    // initialize nodes
    for(int n = 0; n < model.num_nodes; ++n){
        nodes[n] = new Node(n, model, 1);
    }
    
    create_graph(nodes, model);

    // perform actual computation in network
    first_layer_transform(nodes, model.num_nodes, model);
    first_layer_aggregate(nodes, model.num_nodes, model);
    second_layer_transform(nodes, model.num_nodes, model);
    second_layer_aggregate(nodes, model.num_nodes, model);

    // compute accuracy
    float acc = 0.0;
    int pred, correct;
    for(int n = 0; n < model.num_nodes; ++n){
        pred = nodes[n]->get_prediction();
        correct = pred == model.labels[n] ? 1 : 0;
        acc = acc + (float)correct;
    }
    
    acc = acc / model.num_nodes;
    std::cout << "accuracy " << acc << std::endl;
    std::cout << "DONE" << std::endl;

    #if DEBUG
        // for measuring your local runtime
        auto tock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = tock - tick;
        std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
    #endif

    // clean-up 
    for(int n = 0; n < model.num_nodes; ++n){
        nodes[n]->free_node();
        delete nodes[n];
    }

    free(nodes);
    model.free_model();

    (void)argc;
    (void)argv;

    return 0;
}
/***************************************************************************************/