/*
 *
 * File: gcn_mpi.cpp
 * Authors: Batuhan Erden, Yigit Kemal Erinc and Berk Bolgul
 * Created by: Batuhan Erden
 * Created on: Jun 9, 2021
 *
 */

#include <mpi.h>

#include "Model.hpp"
#include "Node.hpp"


#define DEBUG 0


/***************************************************************************************/
/*
 *
 * MPI constants and helper functions
 *
 */

#define TAG_BCAST_STR 999
#define TAG_BCAST_REGULAR 0


void bcast_str(int size, int rank, std::string& str) {
    if (rank == 0) { // Master
		for (int i = 1; i < size; i++) {
            // send str (as char*) to the workers
			MPI_Send(str.c_str(), str.size(), MPI_CHAR, i, TAG_BCAST_STR, MPI_COMM_WORLD);
		}
	} else { // Workers
        // create the str buffer to be received (str will be received as a char* and then casted to string)
        MPI_Status status;
        int str_buffer_length;

        MPI_Probe(0, TAG_BCAST_STR, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_CHAR, &str_buffer_length);

        char* str_buffer = new char[str_buffer_length];

        // receive str (as char*) from the master
		MPI_Recv(str_buffer, str_buffer_length, MPI_CHAR, 0, TAG_BCAST_STR, MPI_COMM_WORLD, &status);

        // cast str to string, and then delete the str buffer
        str = std::string(str_buffer, str_buffer_length);
        delete[] str_buffer;
	}
}
/***************************************************************************************/


/***************************************************************************************/
void create_graph(Node** nodes, Model &model){
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
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_transform(Node* node, Model &model, float* message) {
    for (int c_in = 0; c_in < node->dim_features; ++c_in) {
        float x_in = node->x[c_in];
        float* weight_1_start_idx = model.weight_1 + (c_in * node->dim_hidden);

        for (int c_out = 0; c_out < node->dim_hidden; ++c_out) {
            message[c_out] += x_in * *(weight_1_start_idx + c_out);
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_aggregate(Node** nodes, Model &model, int lower_bound, int upper_bound) {
    Node* node;
    Node* neighbor_node;

    float* message;
    float norm;

    // aggregate for each node
    for (int n = lower_bound; n < upper_bound; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            neighbor_node = nodes[neighbor];
            message = neighbor_node->tmp_hidden; // use already allocated mem

            first_layer_transform(neighbor_node, model, message);

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * neighbor_node->degree);

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
            node->hidden[c] = (node->hidden[c] >= 0.0) * node->hidden[c];
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_transform(Node* node, Model &model, float* message) {
    for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
        float h_in = node->hidden[c_in];
        float* weight_2_start_idx = model.weight_2 + (c_in * node->num_classes);

        for (int c_out = 0; c_out < node->num_classes; ++c_out) {
            message[c_out] += h_in * *(weight_2_start_idx + c_out);
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_aggregate(Node** nodes, Model &model, int lower_bound, int upper_bound) {
    Node* node;
    Node* neighbor_node;

    float* message;
    float norm;

    // aggregate for each node
    for (int n = lower_bound; n < upper_bound; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            neighbor_node = nodes[neighbor];
            message = neighbor_node->tmp_logits; // use already allocated mem

            second_layer_transform(neighbor_node, model, message);

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * neighbor_node->degree);

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
}
/***************************************************************************************/


/***************************************************************************************/
int main(int argc, char** argv) {
    // initialize MPI
    MPI_Init(&argc, &argv);
    int size, rank;

    // determine the size and the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string dataset("");
    int init_no = -1;
    int seed = -1;

    if (rank == 0) { // Master
        // specify problem
        #if DEBUG
            // for measuring your local runtime
            auto tick = std::chrono::high_resolution_clock::now();
            Model::specify_problem(argc, argv, dataset, &init_no, &seed);
        #else
            Model::specify_problem(dataset, &init_no, &seed);
        #endif
    }

    // broadcast dataset
    bcast_str(size, rank, dataset);

    // broadcast init_no and seed
    MPI_Bcast(&init_no, 1, MPI_INT, TAG_BCAST_REGULAR, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_INT, TAG_BCAST_REGULAR, MPI_COMM_WORLD);   

    // load model specifications and model weights
    Model model(dataset, init_no, seed);
    model.load_model();

    // create graph (i.e. load data into each node and load edge structure)
    Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

    if (nodes == nullptr) {
        // finalize MPI
        MPI_Finalize();
        exit(1);
    }

    // initialize nodes
    for (int n = 0; n < model.num_nodes; ++n) {
        nodes[n] = new Node(n, model, 1);
    }

    create_graph(nodes, model);

    // each process has its own data
    const int chunk_size = std::ceil(model.num_nodes / size);
	const int lower_bound = rank * chunk_size; 
	const int upper_bound = std::min(lower_bound + chunk_size, model.num_nodes);

    // perform actual computation in network
    first_layer_aggregate(nodes, model, lower_bound, upper_bound);
    second_layer_aggregate(nodes, model, lower_bound, upper_bound);

    // compute accuracy
    int pred, correct;
    float acc = 0.0;

    for (int n = lower_bound; n < upper_bound; ++n) {
        pred = nodes[n]->get_prediction();
        correct = pred == model.labels[n];

        acc += (float) correct;
    }

    // collect accuracies from all processes and sum them up in the master
    float sum_acc;
    MPI_Reduce(&acc, &sum_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) { // Master
        sum_acc /= model.num_nodes;

        std::cout << "accuracy " << sum_acc << std::endl;
        std::cout << "DONE" << std::endl;

        #if DEBUG
            // for measuring your local runtime
            auto tock = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed_time = tock - tick;
            std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
        #endif
    }

    // clean-up 
    for (int n = 0; n < model.num_nodes; ++n) {
        nodes[n]->free_node();
        delete nodes[n];
    }

    free(nodes);
    model.free_model();

    (void) argc;
    (void) argv;

    // finalize MPI
    MPI_Finalize();

    return 0;
}
/***************************************************************************************/

