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
#define CHUNK_MULTIPLIER 256 // The more this multiplier is, the smaller the chunks will be
#define WORK_KILL_SIGNAL -1 // Once this is received, the workers will stop requesting more work


/***************************************************************************************/
/*
 *
 * MPI tags and helper functions
 *
 */

#define TAG_BCAST_STR 999
#define TAG_WORK_REQUEST 101


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
void first_layer_transform(std::vector<int> neighbor_id_chunk, Node** nodes, Model &model) {
    Node* neighbor_node;

    // transform
    for (int neighbor_id : neighbor_id_chunk) {
        neighbor_node = nodes[neighbor_id];

        for (int c_in = 0; c_in < neighbor_node->dim_features; ++c_in) {
            float x_in = neighbor_node->x[c_in];
            float* weight_1_start_idx = model.weight_1 + (c_in * neighbor_node->dim_hidden);

            for (int c_out = 0; c_out < neighbor_node->dim_hidden; ++c_out) {
                neighbor_node->tmp_hidden[c_out] += x_in * *(weight_1_start_idx + c_out);
            }
        }  
    }
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_aggregate(std::vector<int> node_id_chunk, Node** nodes, Model &model) {
    Node* node;

    float* message;
    float norm;

    // aggregate for each node
    for (int node_id : node_id_chunk) {
        node = nodes[node_id];

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
            node->hidden[c] = (node->hidden[c] >= 0.0) * node->hidden[c];
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_transform(std::vector<int> neighbor_id_chunk, Node** nodes, Model &model) {
    Node* neighbor_node;

    // transform
    for (int neighbor_id : neighbor_id_chunk) {
        neighbor_node = nodes[neighbor_id];

        for (int c_in = 0; c_in < neighbor_node->dim_hidden; ++c_in) {
            float h_in = neighbor_node->hidden[c_in];
            float* weight_2_start_idx = model.weight_2 + (c_in * neighbor_node->num_classes);

            for (int c_out = 0; c_out < neighbor_node->num_classes; ++c_out) {
                neighbor_node->tmp_logits[c_out] += h_in * *(weight_2_start_idx + c_out);
            }
        }  
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_aggregate(std::vector<int> node_id_chunk, Node** nodes, Model &model) {
    Node* node;

    float* message;
    float norm;

    // aggregate for each node
    for (int node_id : node_id_chunk) {
        node = nodes[node_id];

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
}
/***************************************************************************************/


/***************************************************************************************/
void compute_accuracy(std::vector<int> node_id_chunk, Node** nodes, Model& model, float& acc) {
    int pred, correct;

    for (int node_id : node_id_chunk) {
        pred = nodes[node_id]->get_prediction();
        correct = pred == model.labels[node_id];

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
    MPI_Bcast(&init_no, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);   

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

    /* 
     * Dynamic Scheduling:
     * 
     * A one worker might process more nodes than others as we do not know how the graph is built.  
     * For example, if we divided nodes by n chunks and the first chunk had more neighbors than others,
     * the master would process more nodes than the workers and hence be slower.
     * Since other workers would have to wait the master in this case, we use dynamic scheduling!
     */
    // distribute work
    const int chunk_size = std::ceil((float) (model.num_nodes / (CHUNK_MULTIPLIER * size)));

	int start = 0;
    float acc = 0.0;

    if (rank == 0) { // Master
        for (; start < model.num_nodes; start += chunk_size) {
			int curr_worker;

            // receive work request from worker and send work
			MPI_Recv(&curr_worker, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&start, 1, MPI_INT, curr_worker, TAG_WORK_REQUEST, MPI_COMM_WORLD);
		}

		for (int i = 1; i < size; i++) {
			int curr_worker;
            const int work_kill_signal = WORK_KILL_SIGNAL;

			// receive the last work request from worker and send kill signal (no work is left to do!)
			MPI_Recv(&curr_worker, 1, MPI_INT, MPI_ANY_SOURCE, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&work_kill_signal, 1, MPI_INT, curr_worker, TAG_WORK_REQUEST, MPI_COMM_WORLD);
		}
    } else { // Workers
		while (true) {
			// request and receive work from the master
			MPI_Send(&rank, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD);
			MPI_Recv(&start, 1, MPI_INT, 0, TAG_WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// check if the master does not have any more work
			if (start == WORK_KILL_SIGNAL) {
				break;
			}

            const int end = std::min(start + chunk_size, model.num_nodes);

            /*
             * Note: In this implementation, the node ID is equal to the index. So, we can directly use IDs!
             */
            std::vector<int> node_id_chunk, neighbor_id_chunk;

            // fill node_id and neighbor_id chunks
            for (int n = start; n < end; ++n) {
                node_id_chunk.push_back(n);

                for (int neighbor : nodes[n]->neighbors) {
                    neighbor_id_chunk.push_back(neighbor);
                }
            }

			// perform actual computation in network
            first_layer_transform(neighbor_id_chunk, nodes, model);
            first_layer_aggregate(node_id_chunk, nodes, model);
            second_layer_transform(neighbor_id_chunk, nodes, model);
            second_layer_aggregate(node_id_chunk, nodes, model);

            compute_accuracy(node_id_chunk, nodes, model, acc);
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

