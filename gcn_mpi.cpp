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

#define TAG_DATA_COMM_X_VECTOR 102

#define CHUNK_SIZE(num_nodes, size) std::ceil((float) num_nodes / size)
#define END(start, chunk_size, num_nodes) std::min(start + chunk_size, num_nodes)


void bcast_x_vector(const int size, const int rank, const int start, const int end, const int chunk_size, 
                    Node** nodes, Model& model) {
    if (rank == 0) { // Master
        // send X vector to all other processes
        for (int curr_worker = 1; curr_worker < size; ++curr_worker) {
            const int curr_start = curr_worker * chunk_size;
            const int curr_end = END(curr_start, chunk_size, model.num_nodes);

            for (int n = curr_start; n < curr_end; ++n) {
                MPI_Send(nodes[n]->x, model.dim_features, MPI_FLOAT, curr_worker, TAG_DATA_COMM_X_VECTOR, MPI_COMM_WORLD);
            }
        }
    } else { // Workers
        // receive X vector from master
        for (int n = start; n < end; ++n) {
            MPI_Recv(nodes[n]->x, model.dim_features, MPI_FLOAT, 0, TAG_DATA_COMM_X_VECTOR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
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
    Node** nodes = (Node**) malloc(model.num_nodes * sizeof(Node*));

    if (nodes == nullptr) {
        // finalize MPI
        MPI_Finalize();
        exit(1);
    }

    // initialize nodes
    Node* node;

    for (int n = 0; n < model.num_nodes; ++n) {
        node = new Node(n, model, rank == 0); // Only the master reads the X values!
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

float* first_layer_transform(const int chunk_size, const int start, const int end, Node** nodes, Model& model) {
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
                chunk_tmp_hidden[(n - start) * node->dim_hidden + c_out] += tmp_hidden_item;
            }
        }
    }

    return chunk_tmp_hidden;
}

void first_layer_aggregate(Node** nodes, Model &model, float* tmp_hidden, float* full_hidden) {
    Node* node;

    float* message;
    float norm;

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
            full_hidden[n * model.dim_hidden + c] = node->hidden[c];
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
/*
 *
 * Function(s): Second layer operations
 *
 */

float* second_layer_transform(const int chunk_size, const int start, const int end, Node** nodes, Model& model, float* full_hidden) {
    Node* node;

    // tmp_logits for current chunk
    float* chunk_tmp_logits = (float*) calloc(chunk_size * model.num_classes, sizeof(float));

    for (int n = start; n < end; ++n) {
        node = nodes[n];

        for (int c_in = 0; c_in < node->dim_hidden; ++c_in) {
            float h_in = full_hidden[n * model.dim_hidden + c_in];
            float* weight_2_start_idx = model.weight_2 + (c_in * node->num_classes);

            // if the input is zero, do not calculate the corresponding logits
            if (h_in == 0) {
                continue;
            }

            for (int c_out = 0; c_out < node->num_classes; ++c_out) {
                float tmp_logit = h_in * *(weight_2_start_idx + c_out);

                node->tmp_logits[c_out] += tmp_logit;
                chunk_tmp_logits[(n - start) * node->num_classes + c_out] += tmp_logit;
            }
        }
    }

    return chunk_tmp_logits;
}

void second_layer_aggregate(Node** nodes, Model &model, float* tmp_logits) {
    Node* node;

    float* message;
    float norm;

    // aggregate for each node
    for (int n = 0; n < model.num_nodes; ++n) {
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

float compute_accuracy(Node** nodes, Model& model) {
    int pred, correct;
    float acc = 0.0;

    for (int n = 0; n < model.num_nodes; ++n) {
        pred = nodes[n]->get_prediction();
        correct = pred == model.labels[n];

        acc += (float) correct;
    }

    acc /= model.num_nodes;
    return acc;
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

    // distribute work (exluding master since master distributes the work)
    const int chunk_size = CHUNK_SIZE(model.num_nodes, size);
    const int start = rank * chunk_size;
    const int end = END(start, chunk_size, model.num_nodes);

    /*
     * First layer operations
     */

    // broadcast X vector between all processes
    bcast_x_vector(size, rank, start, end, chunk_size, nodes, model);

    // stores the hidden info of each node
    float* full_hidden = (float*) calloc(model.num_nodes * model.dim_hidden, sizeof(float));

    float* tmp_hidden = first_layer_transform(chunk_size, start, end, nodes, model);

    if (rank == 0) { // Master
        // over-allocate in case there are remainders ("chunk_size * size" instead of "model.num_nodes")
        float* tmp_hidden_gathered = (float*) calloc(chunk_size * size * model.dim_hidden, sizeof(float));

        MPI_Gather(tmp_hidden, chunk_size * model.dim_hidden, MPI_FLOAT, 
                   tmp_hidden_gathered, chunk_size * model.dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);

        first_layer_aggregate(nodes, model, tmp_hidden_gathered, full_hidden);
    } else { // Workers
        MPI_Gather(tmp_hidden, chunk_size * model.dim_hidden, MPI_FLOAT, 
                   NULL, chunk_size * model.dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    /*
     * Second layer operations
     */

    // broadcast hidden vector between all processes
    MPI_Bcast(full_hidden, model.num_nodes * model.dim_hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float* tmp_logits = second_layer_transform(chunk_size, start, end, nodes, model, full_hidden);

    if (rank == 0) { // Master
        // over-allocate in case there are remainders ("chunk_size * size" instead of "model.num_nodes")
        float* tmp_logits_gathered = (float*) calloc(chunk_size * size * model.num_classes, sizeof(float));

        MPI_Gather(tmp_logits, chunk_size * model.num_classes, MPI_FLOAT, 
                   tmp_logits_gathered, chunk_size * model.num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);

        second_layer_aggregate(nodes, model, tmp_logits_gathered);
    } else { // Workers
        MPI_Gather(tmp_logits, chunk_size * model.num_classes, MPI_FLOAT, 
                   NULL, chunk_size * model.num_classes, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) { // Master
        float acc = compute_accuracy(nodes, model);

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

