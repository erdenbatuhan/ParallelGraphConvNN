/*
 *
 * Code Owner: TUM Parallel Programming (IN2147) Team
 * Published By: Maximilian Stadler
 *
 */

#ifndef UTILITY_H
#define UTILITY_H

#include <cstdio>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <float.h>
#include <iostream>
#include <string>
#include <queue>
#include <list>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <fstream>
#include <random>
#include <math.h>


class Model {
    public:
        std::string dataset;
        std::string base_dir;
        int init_no;

        float* weight_1;    // weights for first layer
        float* weight_2;    // weights for second layer
        float* bias_1;      // bias for first layer
        float* bias_2;      // bias for second layer

        int* labels;        // GT labels for all nodes
        int* edges;         // edge: node from edges[0][i] to edges[1][i]

        int dim_hidden;     // hidden dimension
        int num_nodes;      // num_nodes
        int dim_features;   // dimension of features
        int num_classes;    // num_classes
        int num_edges;      // num_edges
        int seed;           // seed used for initializing weights randomly

        Model(std::string &d, int i);
        Model(std::string &d, int i, int s);

        void read_meta(void);
        void load_model(void);
        void load_weights(void);
        void initialize_weights(void);
        void free_model(void);

        static char* read_binary(std::string filename, int size, int elem_size);
        static void read_binary_into_buffer(char* buffer, std::string filename, int size, int elem_size);
        static void write_binary(std::string filename, char* buffer, int size, int elem_size);

        static void validate_input(std::string &dataset, int init_no, int seed);
        static void specify_problem(std::string &dataset, int* init_no, int* seed);
        static void specify_problem(int argc, char** argv, std::string &dataset, int* init_no, int* seed);
};


#endif //UTILITY_H
