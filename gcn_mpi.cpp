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
 * MPI constants and functions
 *
 */

#define TAG_WORK_REQUEST 101
#define TAG_DATA_REQUEST 102

#define CHUNK_MULTIPLIER 8 // The more this multiplier is, the smaller the chunks will be
#define WORK_KILL_SIGNAL -1 // Once this is received, the workers will stop requesting more work

#define CHUNK_SIZE(num_nodes, size) std::ceil((float) num_nodes / (CHUNK_MULTIPLIER * size))
#define END(start, chunk_size, num_nodes) std::min(start + chunk_size, num_nodes)


void provide_work_to_workers(int* start, const int chunk_size, Node** nodes, Model& model) { // Master
    int curr_worker;
    const int end = END(*(start), chunk_size, model.num_nodes);

    // receive work request from worker and send work
    MPI_Recv(&curr_worker, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(start, 1, MPI_INT, curr_worker, TAG_WORK_REQUEST, MPI_COMM_WORLD);

    // receive data request from worker
    MPI_Recv(&curr_worker, 1, MPI_INT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // send data to worker (for each neighbor, send the corresponding X vector)
    for (int n = *(start); n < end; ++n) {
        for (int neighbor : nodes[n]->neighbors) {
            MPI_Send(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        }
    }
}

void send_kill_signal_to_workers(const int size) { // Master
    for (int i = 1; i < size; i++) {
        int curr_worker;
        const int work_kill_signal = WORK_KILL_SIGNAL;

        // receive the last work request from worker and send kill signal (no work is left to do!)
        MPI_Recv(&curr_worker, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&work_kill_signal, 1, MPI_INT, curr_worker, TAG_WORK_REQUEST, MPI_COMM_WORLD);
    }
}

bool request_and_receive_work_from_master(const int rank, int* start) { // Workers
    // request and receive work from the master
    MPI_Send(&rank, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD);
    MPI_Recv(start, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // check if the master does not have any more work
    return *start != WORK_KILL_SIGNAL;
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Creating and loading the model
 *
 */

Model load_model(const int rank, std::string dataset, int init_no, int seed) {
    Model model(dataset, init_no, seed);

    // (Master) load model specifications and model weights
    if (rank == 0) { // Master
        model.load_model();
    }

    // broadcast model attributes
    int num_nodes, num_edges, dim_features, dim_hidden, num_classes;

    if (rank == 0) { // Master
        num_nodes = model.num_nodes;
        num_edges = model.num_edges;
        dim_features = model.dim_features;
        dim_hidden = model.dim_hidden;
        num_classes = model.num_classes;
    }

    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_hidden, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float* weight_1 = (float*) calloc(dim_features * dim_hidden, sizeof(float));
    float* weight_2 = (float*) calloc(dim_hidden * num_classes, sizeof(float));
    float* bias_1 = (float*) calloc(dim_hidden, sizeof(float));
    float* bias_2 = (float*) calloc(num_classes, sizeof(float));
    int* labels = (int*) calloc(num_nodes, sizeof(int));
    int* edges = (int*) calloc(num_edges * 2, sizeof(int));

    if (rank == 0) { // Master
        weight_1 = model.weight_1;
        weight_2 = model.weight_2;
        bias_1 = model.bias_1;
        bias_2 = model.bias_2;
        labels = model.labels;
        edges = model.edges;
    }

    MPI_Bcast(weight_1, dim_features * dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight_2, dim_hidden * num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_1, dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_2, num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(labels, num_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(edges, num_edges * 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    // (Workers) set model attributes broadcasted
    if (rank != 0) {
        model.num_nodes = num_nodes;
        model.num_edges = num_edges;
        model.dim_features = dim_features;
        model.dim_hidden = dim_hidden;
        model.num_classes = num_classes;
        model.weight_1 = weight_1;
        model.weight_2 = weight_2;
        model.bias_1 = bias_1;
        model.bias_2 = bias_2;
        model.labels = labels;
        model.edges = edges;
    }
    
    return model;
}

Model create_model(const int rank) {
    // initialize the parameters needed for model creation
    std::string dataset("");
    int init_no = -1;
    int seed = -1;

    // read input to specify problem, then load model specifications and model weights
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

    // load or receive model
    return load_model(rank, dataset, init_no, seed);
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Creating the graph
 *
 */

Node** create_nodes(const int rank, Model& model) {
    // create graph (i.e. load data into each node and load edge structure)
    Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

    if (nodes == nullptr) {
        // finalize MPI
        MPI_Finalize();
        exit(1);
    }

    // initialize nodes
    for (int n = 0; n < model.num_nodes; ++n) {
        Node* node = new Node(n, model, rank == 0); // Only the master reads the X values!
        nodes[n] = node;

        if (rank != 0) { // Workers
            node->tmp_hidden = (float*) calloc(node->dim_hidden, sizeof(float));
            node->hidden = (float*) calloc(node->dim_hidden, sizeof(float));
            node->tmp_logits = (float*) calloc(node->num_classes, sizeof(float));
            node->logits = (float*) calloc(node->num_classes, sizeof(float));
            node->x = (float*) calloc(model.dim_features, sizeof(float));
        }
    }

    return nodes;
}

void create_graph(Node** nodes, Model &model) {
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
/*
 *
 * Function(s): First layer operations
 *
 */

float* first_layer_transform(Node* node, float* weight_1) {
    // transform
    for (int c_in = 0; c_in < node->dim_features; ++c_in) {
        float x_in = node->x[c_in];
        float* weight_1_start_idx = weight_1 + (c_in * node->dim_hidden);

        // if the input is zero, do not calculate the corresponding hidden values
        if (x_in == 0) {
            continue;
        }

        for (int c_out = 0; c_out < node->dim_hidden; ++c_out) {
            node->tmp_hidden[c_out] += x_in * *(weight_1_start_idx + c_out);
        }
    }

    return node->tmp_hidden;
}

bool* first_layer_aggregate(const int start, const int end, Node** nodes, Model &model) {
    Node* node;
    Node* neighbor_node;

    float* message;
    float norm;

    // keep the processed nodes and neighbors
    bool* processed_nodes = (bool*) malloc(model.num_nodes * sizeof(bool));
    bool* processed_neighbors = (bool*) malloc(model.num_nodes * sizeof(bool));

    // weight
    float* weight_1 = model.weight_1;

    // aggregate for each node
    for (int n = start; n < end; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            neighbor_node = nodes[neighbor];
            message = neighbor_node->tmp_hidden;

            // transform if not processed before
            if (!processed_neighbors[neighbor]) {
                message = first_layer_transform(neighbor_node, weight_1);
                processed_neighbors[neighbor] = true; // set processed to true
            }

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

        processed_nodes[n] = true; // set processed to true
    }

    return processed_nodes;
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Second layer operations
 *
 */

float* second_layer_transform(Node* node, float* weight_2) {
    // transform
    for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
        float h_in = node->hidden[c_in];
        float* weight_2_start_idx = weight_2 + (c_in * node->num_classes);

        // if the input is zero, do not calculate the corresponding logits
        if (h_in == 0) {
            continue;
        }

        for (int c_out = 0; c_out < node->num_classes; ++c_out) {
            node->tmp_logits[c_out] += h_in * *(weight_2_start_idx + c_out);
        }
    }

    return node->tmp_logits;
}

void second_layer_aggregate(const int start, const int end, Node** nodes, Model &model, 
                            bool* first_layer_processed_nodes) {
    Node* node;
    Node* neighbor_node;

    float* message;
    float norm;

    // keep the processed neighbors
    bool* processed_neighbors = (bool*) malloc(model.num_nodes * sizeof(bool));

    // weight
    float* weight_2 = model.weight_2;

    // aggregate for each node
    for (int n = start; n < end; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            // continue if not processed before (in the first layer)
            if (!first_layer_processed_nodes[neighbor]) {
                continue;
            }

            neighbor_node = nodes[neighbor];
            message = neighbor_node->tmp_logits;
            
            // transform if not processed before (in the current layer)
            if (!processed_neighbors[neighbor]) {
                message = second_layer_transform(neighbor_node, weight_2);
                processed_neighbors[neighbor] = true; // set processed to true
            }

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
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Computing the accuracy
 *
 */

void compute_accuracy(const int start, const int end, Node** nodes, Model& model, float& acc) {
    int pred, correct;

    for (int n = start; n < end; ++n) {
        pred = nodes[n]->get_prediction();
        correct = pred == model.labels[n];

        acc += (float) correct;
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

    // create model (master reads, workers receive!)
    Model model = create_model(rank);

    // create graph (only the master loads data and edge structure!)
    Node** nodes = create_nodes(rank, model);
    create_graph(nodes, model);

    /* 
     * Dynamic Scheduling:
     * 
     * A one worker might process more nodes than others as we do not know how the graph is built.  
     * For example, if we divided nodes by n chunks and the first chunk had more neighbors than others,
     * the master would process more nodes than the workers and hence be slower.
     * Since other workers would have to wait the master in this case, we use dynamic scheduling!
     */
    // distribute work (exluding master since master distributes the work)
    const int chunk_size = CHUNK_SIZE(model.num_nodes, (size - 1));

    int start = 0;
    float acc = 0.0;

    if (rank == 0) { // Master
        for (; start < model.num_nodes; start += chunk_size) {
            provide_work_to_workers(&start, chunk_size, nodes, model);
        }

        send_kill_signal_to_workers(size);
    } else { // Workers
		while (request_and_receive_work_from_master(rank, &start)) {
            MPI_Send(&rank, 1, MPI_INT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD);

            const int end = END(start, chunk_size, model.num_nodes);

            for (int n = start; n < end; ++n) {
                for (int neighbor : nodes[n]->neighbors) {
                    MPI_Recv(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // perform actual computation in network
            bool* first_layer_processed_nodes = first_layer_aggregate(start, end, nodes, model);
            second_layer_aggregate(start, end, nodes, model, first_layer_processed_nodes);

            compute_accuracy(start, end, nodes, model, acc);
		}
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

