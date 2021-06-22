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
#define TAG_RESULT_REQUEST 103

#define CHUNK_MULTIPLIER 1 // The more this multiplier is, the smaller the chunks will be
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
        //MPI_Send(&nodes[n]->degree, 1, MPI_INT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        //MPI_Send(&nodes[n]->neighbors[0], nodes[n]->degree, MPI_INT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);

        // for (int neighbor : nodes[n]->neighbors) {
        //     // MPI_Send(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        //     //MPI_Send(nodes[neighbor]->tmp_hidden, model.dim_hidden, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        //     //MPI_Send(nodes[neighbor]->tmp_logits, model.num_classes, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        // }

        MPI_Send(nodes[n]->x, model.dim_features, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);

        //MPI_Send(nodes[n]->hidden, model.dim_hidden, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
        //MPI_Send(nodes[n]->logits, model.num_classes, MPI_FLOAT, curr_worker, TAG_DATA_REQUEST, MPI_COMM_WORLD);
    }

    // // get results from the workers
    // for (int n = *(start); n < end; ++n) {
    //     //for (int neighbor : nodes[n]->neighbors) {
    //         //MPI_Recv(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, curr_worker, TAG_RESULT_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         //MPI_Recv(nodes[neighbor]->tmp_hidden, model.dim_hidden, MPI_FLOAT, curr_worker, TAG_RESULT_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         //MPI_Recv(nodes[neighbor]->tmp_logits, model.num_classes, MPI_FLOAT, curr_worker, TAG_RESULT_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     //}

    //     MPI_Recv(nodes[n]->hidden, model.dim_hidden, MPI_FLOAT, curr_worker, TAG_RESULT_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     //MPI_Recv(nodes[n]->logits, model.num_classes, MPI_FLOAT, curr_worker, TAG_RESULT_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }
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

bool receive_work_from_master(const int rank, int* start, const int chunk_size, Node** nodes, Model& model) { // Workers
    // request and receive work from the master
    MPI_Send(&rank, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD);
    MPI_Recv(start, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // check if the master does not have any more work left to do
    if (*(start) == WORK_KILL_SIGNAL) {
        return false;
    }

    // send data request to worker
    MPI_Send(&rank, 1, MPI_INT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD);

    const int end = END(*(start), chunk_size, model.num_nodes);

    // receive data from worker (for each neighbor, receive the corresponding X vector)
    for (int n = *(start); n < end; ++n) {
        //MPI_Recv(&nodes[n]->degree, 1, MPI_INT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //nodes[n]->neighbors.resize(nodes[n]->degree);
        //MPI_Recv(&nodes[n]->neighbors[0], nodes[n]->degree, MPI_INT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // for (int neighbor : nodes[n]->neighbors) {
        //     MPI_Recv(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     //MPI_Recv(nodes[neighbor]->tmp_hidden, model.dim_hidden, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     //MPI_Recv(nodes[neighbor]->tmp_logits, model.num_classes, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }

        MPI_Recv(nodes[n]->x, model.dim_features, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //MPI_Recv(nodes[n]->hidden, model.dim_hidden, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(nodes[n]->logits, model.num_classes, MPI_FLOAT, 0, TAG_DATA_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return true;
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
    std::string dataset("COAUTHORCS");
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
    Node* node;

    for (int n = 0; n < model.num_nodes; ++n) {
        node = new Node(n, model, 1); // Only the master reads the X values!
        nodes[n] = node;

        /*
        if (rank != 0) { // Workers
            node->tmp_hidden = (float*) calloc(node->dim_hidden, sizeof(float));
            node->hidden = (float*) calloc(node->dim_hidden, sizeof(float));
            node->tmp_logits = (float*) calloc(node->num_classes, sizeof(float));
            node->logits = (float*) calloc(node->num_classes, sizeof(float));
            node->x = (float*) calloc(model.dim_features, sizeof(float));
        }

        MPI_Bcast(node->x, model.dim_features, MPI_FLOAT, 0, MPI_COMM_WORLD);*/
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

float* first_layer_transform(const int start, const int end, const int chunk_size, Node** nodes, Model& model) {
    Node* node;

    // tmp_hidden for current chunk
    float* chunk_tmp_hidden = (float*) calloc(chunk_size * model.dim_hidden, sizeof(float));

    for (int n = start; n < end; ++n) {
        node = nodes[n];

        for (int c_in = 0; c_in < node->dim_features; ++c_in) {
            float x_in = node->x[c_in];
            float* weight_1_start_idx = model.weight_1 + (c_in * node->dim_hidden);

            // if the input is zero, do not calculate the corresponding hidden values
            if (x_in == 0) {
                continue;
            }

            for (int c_out = 0; c_out < node->dim_hidden; ++c_out) {
                float tmp_hidden_item = x_in * *(weight_1_start_idx + c_out);

                node->tmp_hidden[c_out] += tmp_hidden_item;
                chunk_tmp_hidden[(n - start) * model.dim_hidden + c_out] += tmp_hidden_item;
            }
        }
    }

    return chunk_tmp_hidden;
}

float* first_layer_aggregate(const int start, const int end, Node** nodes, Model &model, float* tmp_hidden) {
    Node* node;

    float* message;
    float norm;

    float* hidden_vector = (float*) calloc(model.num_nodes * model.dim_hidden, sizeof(float));

    // aggregate for each node
    for (int n = 0; n < model.num_nodes; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            message = &tmp_hidden[neighbor * model.dim_hidden];

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message and add bias
            for (int c = 0; c < node->dim_hidden; ++c) {
                node->hidden[c] += message[c] / norm + model.bias_1[c] / node->degree;
            }
        }

        // apply relu
        for (int c = 0; c < node->dim_hidden; ++c) {
            node->hidden[c] = (node->hidden[c] >= 0.0) * node->hidden[c];
            hidden_vector[n * model.dim_hidden + c] = node->hidden[c];
        }
    }

    return hidden_vector;
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Second layer operations
 *
 */

float* second_layer_transform(const int start, const int end, const int chunk_size, Node** nodes, Model& model) {
    Node* node;

    // tmp_logits for current chunk
    float* chunk_tmp_logits = (float*) calloc(chunk_size * model.num_classes, sizeof(float));

    for (int n = start; n < end; ++n) {
        node = nodes[n];

        for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
            float h_in = node->hidden[c_in];
            float* weight_2_start_idx = model.weight_2 + (c_in * node->num_classes);

            // if the input is zero, do not calculate the corresponding logits
            if (h_in == 0) {
                continue;
            }

            for (int c_out = 0; c_out < node->num_classes; ++c_out) {
                float tmp_logit = h_in * *(weight_2_start_idx + c_out);

                node->tmp_logits[c_out] += tmp_logit;
                chunk_tmp_logits[(n - start) * model.num_classes + c_out] += tmp_logit;
            }
        }
    }

    return chunk_tmp_logits;
}

void second_layer_aggregate(const int start, const int end, Node** nodes, Model &model, float* tmp_logits) {
    Node* node;

    float* message;
    float norm;

    // aggregate for each node
    for (int n = start; n < end; ++n) {
        node = nodes[n];

        // aggregate from each neighbor
        for (int neighbor : node->neighbors) {
            message = &tmp_logits[neighbor * model.num_classes];

            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message and add bias
            for (int c = 0; c < node->num_classes; ++c) {
                node->logits[c] += message[c] / norm + model.bias_2[c] / node->degree;
            }
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


    auto tick = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < model.num_nodes; ++n) {
        for (int neighbor1 : nodes[n]->neighbors) {
            for (int neighbor2 : nodes[neighbor1]->neighbors) {
                //
            }
        }
    }
    auto tock = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_time = tock - tick;
    std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;

    /* 
     * Dynamic Scheduling:
     * 
     * A one worker might process more nodes than others as we do not know how the graph is built.  
     * For example, if we divided nodes by n chunks and the first chunk had more neighbors than others,
     * the master would process more nodes than the workers and hence be slower.
     * Since other workers would have to wait the master in this case, we use dynamic scheduling!
     */
    // distribute work (exluding master since master distributes the work)
    const int chunk_size = CHUNK_SIZE(model.num_nodes, size);

    const int start = rank * chunk_size;
    const int end = END(start, chunk_size, model.num_nodes);

    float* tmp_hidden = first_layer_transform(start, end, chunk_size, nodes, model);
    float* tmp_hidden_gathered = (float*) calloc(model.num_nodes * model.dim_hidden, sizeof(float));

    MPI_Allgather(tmp_hidden, chunk_size * model.dim_hidden, MPI_FLOAT, 
                  tmp_hidden_gathered, chunk_size * model.dim_hidden, MPI_FLOAT, MPI_COMM_WORLD);
    first_layer_aggregate(start, end, nodes, model, tmp_hidden_gathered);

    float* tmp_logits = second_layer_transform(start, end, chunk_size, nodes, model);
    float* tmp_logits_gathered = (float*) calloc(model.num_nodes * model.num_classes, sizeof(float));

    MPI_Allgather(tmp_logits, chunk_size * model.num_classes, MPI_FLOAT, 
                  tmp_logits_gathered, chunk_size * model.num_classes, MPI_FLOAT, MPI_COMM_WORLD);
    second_layer_aggregate(start, end, nodes, model, tmp_logits_gathered);

    float acc = 0.0;
    compute_accuracy(start, end, nodes, model, acc);

    // for (int p = 0; p < 1; ++p) {
    //     if (rank == 0) { // Master
    //         for (; start < model.num_nodes; start += chunk_size) {
    //             provide_work_to_workers(&start, chunk_size, nodes, model);

    //             if (p == 0) {

    //             }
    //         }

    //         send_kill_signal_to_workers(size);
    //     } else { // Workers
    //         while (receive_work_from_master(rank, &start, chunk_size, nodes, model)) {
    //             const int end = END(start, chunk_size, model.num_nodes);

    //             // perform actual computation in network
    //             float* local_tmp_hidden = first_layer_transform(start, end, chunk_size, nodes, model);

    //             MPI_Allgather(local_tmp_hidden, chunk_size * model.dim_hidden, MPI_FLOAT, 
    //                           tmp_hidden, chunk_size * model.dim_hidden, MPI_FLOAT, MPI_COMM_WORLD);

    //             // compute_accuracy(start, end, nodes, model, acc);

    //             // // provide master with results
    //             // for (int n = start; n < end; ++n) {
    //             //     //for (int neighbor : nodes[n]->neighbors) {
    //             //         //MPI_Send(nodes[neighbor]->x, model.dim_features, MPI_FLOAT, 0, TAG_RESULT_REQUEST, MPI_COMM_WORLD);
    //             //         //MPI_Send(nodes[neighbor]->tmp_hidden, model.dim_hidden, MPI_FLOAT, 0, TAG_RESULT_REQUEST, MPI_COMM_WORLD);
    //             //         //MPI_Send(nodes[neighbor]->tmp_logits, model.num_classes, MPI_FLOAT, 0, TAG_RESULT_REQUEST, MPI_COMM_WORLD);
    //             //     //}

    //             // MPI_Send(nodes[n]->hidden, model.dim_hidden, MPI_FLOAT, 0, TAG_RESULT_REQUEST, MPI_COMM_WORLD);
    //             // //MPI_Send(nodes[n]->logits, model.num_classes, MPI_FLOAT, 0, TAG_RESULT_REQUEST, MPI_COMM_WORLD);
    //             // }
    //         }
    //     }
    // }

    // collect accuracies from all processes and sum them up in the master
    //float sum_acc;
    //MPI_Reduce(&acc, &sum_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) { // Master
        acc /= model.num_nodes;

        std::cout << "accuracy " << acc << std::endl;
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

