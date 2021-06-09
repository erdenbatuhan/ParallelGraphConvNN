/*
 *
 * File: Node.cpp
 * Authors: TUM Parallel Programming (IN2147) Team
 * Published by: Maximilian Stadler
 * Published on: Jun 8, 2021
 *
 */

#include "Node.hpp"


// constructor for node
Node::Node(int id, Model &model, bool allocate_memory) {
    this->ID = id;
    this->dim_features = model.dim_features;
    this->dim_hidden = model.dim_hidden;
    this->num_classes = model.num_classes;

    if(allocate_memory){
        this->tmp_hidden = (float*)calloc(this->dim_hidden, sizeof(float));
        this->hidden = (float*)calloc(this->dim_hidden, sizeof(float));
        this->tmp_logits = (float*)calloc(this->num_classes, sizeof(float));
        this->logits = (float*)calloc(this->num_classes, sizeof(float));

        std::stringstream ss; ss << model.base_dir << "X/x_" << id << ".bin";
        this->x = (float*)Model::read_binary(ss.str(), model.dim_features, sizeof(float));

        if(this->x == nullptr){
            exit(1);
        }
    } else {
        this->x = nullptr;
        this->tmp_hidden = nullptr;
        this->hidden = nullptr;
        this->logits = nullptr;
        this->tmp_logits = nullptr;
    }
}


// free dynamic memory allocated in node
void Node::free_node(void){
    free(this->x);
    free(this->tmp_hidden);
    free(this->hidden);
    free(this->tmp_logits);
    free(this->logits);
}


// extract prediction (i.e. argmax)
int Node::get_prediction(void){
    float y_max = this->logits[0];
    int c_max = 0;
    for(int c = 1; c < this->num_classes; ++c){
        if (this->logits[c] > y_max){
            y_max = this->logits[c];
            c_max = c;
        }
    }
    this->y = c_max;
    return this->y;
}