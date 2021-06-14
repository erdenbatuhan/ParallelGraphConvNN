/*
 *
 * File: gcn_omp.cpp
 * Authors: Batuhan Erden, Yigit Kemal Erinc and Berk Bolgul
 * Created by: Batuhan Erden
 * Created on: Jun 9, 2021
 *
 */

#include "Model.hpp"
#include "Node.hpp"

#include <omp.h>


#define DEBUG 0
#define PRINT_TIME 1


/***************************************************************************************/
void create_graph(Node** nodes, Model& model, std::chrono::duration<double>& time_passed) {
    auto tick = std::chrono::high_resolution_clock::now();

    // set neighbor relations
    int source, target;

    for (int e = 0; e < model.num_edges; ++e) {
        source = model.edges[e];
        target = model.edges[model.num_edges + e];

        // self-loops twice in edges, so ignore for now
        // and add later
        if (source != target) {
            nodes[source]->neighbors.push_back(target);
        }
    }

    // add self-loops
    for (int n = 0; n < model.num_nodes; ++n) {
        Node *node = nodes[n];

        node->neighbors.push_back(node->ID);
        node->degree = node->neighbors.size();
    }

    auto tock = std::chrono::high_resolution_clock::now();
    time_passed = tock - tick;
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_transform(Node** nodes, int num_nodes, Model& model, std::chrono::duration<double>& time_passed) {
    auto tick = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_nodes; ++i) {
        Node* node = nodes[i];

        for (int c_out = 0; c_out < node->dim_hidden; ++c_out) {
            for (int c_in = 0; c_in < node->dim_features; ++c_in) {
                node->tmp_hidden[c_out] += node->x[c_in] * model.weight_1[c_in * node->dim_hidden + c_out];
            }
        }
    }

    auto tock = std::chrono::high_resolution_clock::now();
    time_passed = tock - tick;
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_aggregate(Node** nodes, int num_nodes, Model& model, std::chrono::duration<double>& time_passed) {
    auto tick = std::chrono::high_resolution_clock::now();

    // aggregate
    float* message;
    float norm;
    Node* node;

    for (int n = 0; n < num_nodes; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            message = nodes[neighbor]->tmp_hidden;

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message
            for (int c = 0; c < node->dim_hidden; ++c) {
                node->hidden[c] += message[c] / norm;
            }
        }

        // add bias
        for (int c = 0; c < node->dim_hidden; ++c) {
            node->hidden[c] += model.bias_1[c];
        }

        // apply relu
        for (int c = 0; c < node->dim_hidden; ++c) {
            node->hidden[c] = node->hidden[c] < 0.0 ? 0.0 : node->hidden[c];
        }
    }

    auto tock = std::chrono::high_resolution_clock::now();
    time_passed = tock - tick;
}
/***************************************************************************************/


/***************************************************************************************/
// computation in second layer
void second_layer_transform(Node** nodes, int num_nodes, Model& model, std::chrono::duration<double>& time_passed) {
    auto tick = std::chrono::high_resolution_clock::now();

    // transform
    Node* node;

    for (int n = 0; n < num_nodes; ++n) {
        node = nodes[n];

        for (int c_out = 0; c_out < node->num_classes; ++c_out) {
            for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
                node->tmp_logits[c_out] += node->hidden[c_in] * model.weight_2[c_in * node->num_classes + c_out];
            }
        }   
    }

    auto tock = std::chrono::high_resolution_clock::now();
    time_passed = tock - tick;
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_aggregate(Node** nodes, int num_nodes, Model& model, std::chrono::duration<double>& time_passed) {
    auto tick = std::chrono::high_resolution_clock::now();

    // aggregate
    Node* node;

    float* message;
    float norm;

    // for each node
    for (int n = 0; n < num_nodes; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            message = nodes[neighbor]->tmp_logits;

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message
            for (int c = 0; c < node->num_classes; ++c) {
                node->logits[c] += message[c] / norm;
            }
        }

        // add bias
        for (int c = 0; c < node->num_classes; ++c) {
            node->logits[c] += model.bias_2[c];
        }
    }

    auto tock = std::chrono::high_resolution_clock::now();
    time_passed = tock - tick;      
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

    auto tick_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_passed_create_graph;
    std::chrono::duration<double> time_passed_first_layer_transform;
    std::chrono::duration<double> time_passed_first_layer_aggregate;
    std::chrono::duration<double> time_passed_second_layer_transform;
    std::chrono::duration<double> time_passed_second_layer_aggregate;

    // load model specifications and model weights
    Model model(dataset, init_no, seed);
    model.load_model();

    // create graph (i.e. load data into each node and load edge structure)
    Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

    if (nodes == nullptr) {
        exit(1);
    }

    // initialize nodes
    for (int n = 0; n < model.num_nodes; ++n) {
        nodes[n] = new Node(n, model, 1);
    }
    
    create_graph(nodes, model, time_passed_create_graph);

    // perform actual computation in network
    first_layer_transform(nodes, model.num_nodes, model, time_passed_first_layer_transform);
    first_layer_aggregate(nodes, model.num_nodes, model, time_passed_first_layer_aggregate);
    second_layer_transform(nodes, model.num_nodes, model, time_passed_second_layer_transform);
    second_layer_aggregate(nodes, model.num_nodes, model, time_passed_second_layer_aggregate);

    // compute accuracy
    int pred, correct;
    float acc = 0.0;

    for (int n = 0; n < model.num_nodes; ++n) {
        pred = nodes[n]->get_prediction();
        correct = pred == model.labels[n];

        acc += (float) correct;
    }
    
    acc /= model.num_nodes;

    std::cout << "accuracy " << acc << std::endl;
    std::cout << "DONE" << std::endl;

    #if DEBUG
        // for measuring your local runtime
        auto tock = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_time = tock - tick;
        std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
    #endif

    // clean-up 
    for (int n = 0; n < model.num_nodes; ++n) {
        nodes[n]->free_node();
        delete nodes[n];
    }

    free(nodes);
    model.free_model();

    (void) argc;
    (void) argv;

    auto tock_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_passed = tock_total - tick_total;

    #if PRINT_TIME
        std::cout << "(Total) time passed: " << time_passed.count() << std::endl;
        std::cout << "(create_graph) time passed: " << time_passed_create_graph.count() << ", " << (time_passed_create_graph.count() / time_passed.count()) * 100 << "%" << std::endl;
        std::cout << "(first_layer_transform) time passed: " << time_passed_first_layer_transform.count() << ", " << (time_passed_first_layer_transform.count() / time_passed.count()) * 100 << "%" << std::endl;
        std::cout << "(first_layer_aggregate) time passed: " << time_passed_first_layer_aggregate.count() << ", " << (time_passed_first_layer_aggregate.count() / time_passed.count()) * 100 << "%" << std::endl;
        std::cout << "(second_layer_transform) time passed: " << time_passed_second_layer_transform.count() << ", " << (time_passed_second_layer_transform.count() / time_passed.count()) * 100 << "%" << std::endl;
        std::cout << "(second_layer_aggregate) time passed: " << time_passed_second_layer_aggregate.count() << ", " << (time_passed_second_layer_aggregate.count() / time_passed.count()) * 100 << "%" << std::endl;
    #endif

    return 0;
}
/***************************************************************************************/

