#include "Model.hpp"


Model::Model(std::string &d, int i) 
    : init_no{i}, 
      weight_1{nullptr}, weight_2{nullptr}, bias_1{nullptr}, bias_2{nullptr},
      labels{nullptr}, edges{nullptr}, 
      dim_hidden{0}, num_nodes{0}, dim_features{0}, num_classes{0},
      num_edges{0}, seed{42} {

    this->base_dir.append("./data/").append(d).append("/");
}


Model::Model(std::string &d, int i, int s) 
    : init_no{i}, 
      weight_1{nullptr}, weight_2{nullptr}, bias_1{nullptr}, bias_2{nullptr},
      labels{nullptr}, edges{nullptr}, 
      dim_hidden{0}, num_nodes{0}, dim_features{0}, num_classes{0},
      num_edges{0}, seed{s} {
    
    this->base_dir.append("./data/").append(d).append("/");
}


void Model::read_meta(void){
    // file-name
    std::string path("");
    path.append(this->base_dir).append("meta.txt");

    // open file
    std::ifstream file(path);
    std::string meta = "";

    if (file.fail()){
        std::cerr << "File " << path << " does not exist, aborting!" << std::endl;
        exit(1);
    }

    // read file
    file.seekg(0, std::ios::end);   
    meta.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    meta.assign((std::istreambuf_iterator<char>(file)),
                 std::istreambuf_iterator<char>());

    // extract file content
    std::smatch match;

    // num_classes
    std::regex num_classes("num_classes: (\\d+)\n");
    std::regex_search(meta, match, num_classes);
    this->num_classes = std::stoi(match[1].str().c_str());

    // dim_features
    std::regex dim_features("dim_features: (\\d+)\n");
    std::regex_search(meta, match, dim_features);
    this->dim_features = std::stoi(match[1].str().c_str());

    // hidden dim
    std::regex dim_hidden("dim_hidden: (\\d+)\n");
    std::regex_search(meta, match, dim_hidden);
    this->dim_hidden = std::stoi(match[1].str().c_str());

    // num_nodes
    std::regex num_nodes("num_nodes: (\\d+)\n");
    std::regex_search(meta, match, num_nodes);
    this->num_nodes = std::stoi(match[1].str().c_str());

    // num_nodes
    std::regex num_edges("num_edges: (\\d+)\n");
    std::regex_search(meta, match, num_edges);
    this->num_edges = std::stoi(match[1].str().c_str());
}



char* Model::read_binary(std::string filename, int size, int elem_size){
    std::ifstream binary(filename, std::ios::binary);
    if (binary.fail()){
        std::cerr << "File " << filename << " does not exist, aborting!" << std::endl;
        exit(1);
    }

    char *buffer = (char*)calloc(size, elem_size);
    if(buffer == nullptr){
        exit(1);
    }
    binary.read(buffer, size * elem_size);

    return buffer;
}


void Model::read_binary_into_buffer(char* buffer, std::string filename, int size, int elem_size){
    std::ifstream binary(filename, std::ios::binary);
    if (binary.fail()){
        std::cerr << "File " << filename << " does not exist, aborting!" << std::endl;
        exit(1);
    }
    binary.read(buffer, size * elem_size);
}



void Model::write_binary(std::string filename, char* buffer, int size, int elem_size){
    std::ofstream(filename, std::ios::binary).write(buffer, size * elem_size);
}


void Model::load_model(void){
    // read meta information defining the model's architecture
    this->read_meta();

    // load labels
    std::stringstream path;
    path << this->base_dir;
    std::string filename = path.str().append("y.bin");
    this->labels = (int*)Model::read_binary(filename, this->num_nodes, sizeof(int));

    // load edges
    filename = path.str().append("edges.bin");
    this->edges = (int*)Model::read_binary(filename, this->num_edges * 2, sizeof(int));

    // if init-no < 0, i.e. random weights
    if(this->init_no < 0){
        this->initialize_weights();
    }
    else {
        this->load_weights();
    }
}


void Model::initialize_weights(void){
    std::mt19937 random(this->seed);

    // Kaiming initialization for weights, i.e. U(-bound, bound), where bound = sqrt(3.0 / in_channels)
    // i.e. initializations depends on size of previous layer
    this->weight_1 = (float*)calloc(this->dim_features * this->dim_hidden, sizeof(float));
    this->weight_2 = (float*)calloc(this->dim_hidden * this->num_classes, sizeof(float));

    if(weight_1 == nullptr){
        exit(1);
    }
    if(weight_2 == nullptr){
        exit(1);
    }
    
    float bound_w1 = sqrt(3.0 / (this->dim_features));
    float bound_w2 = sqrt(3.0 / (this->dim_hidden));
    std::uniform_real_distribution<float> dist_w1(-bound_w1, bound_w1);
    std::uniform_real_distribution<float> dist_w2(-bound_w2, bound_w2);

    for(int w = 0; w < this->dim_features * this->dim_hidden; ++w){
        this->weight_1[w] = dist_w1(random);
    }

    for(int w = 0; w < this->dim_hidden * this->num_classes; ++w){
        this->weight_2[w] = dist_w2(random);
    }
    // bias-terms are initialized to small random perturbations
    // usually they are initialized to 0, but this would lead to making computations in that regard obsolete
    this->bias_1 = (float*)calloc(this->dim_hidden, sizeof(float));
    this->bias_2 = (float*)calloc(this->num_classes, sizeof(float));

    if(bias_1 == nullptr){
        exit(1);
    }
    if(bias_2 == nullptr){
        exit(1);
    }

    std::normal_distribution<float> normal(0, 0.001);
    for(int b = 0; b < this->dim_hidden; ++b){
        this->bias_1[b] = normal(random);
    }
    for(int b = 0; b < this->num_classes; ++b){
        this->bias_2[b] = normal(random);
    }
}


void Model::load_weights(void){
    // load weights and biases
    std::stringstream weight_dir;
    weight_dir << this->base_dir << this->init_no << "/"; 
    
    // load weights of first layer
    std::string filename = weight_dir.str().append("weight_1.bin");
    this->weight_1 = (float*)Model::read_binary(filename, this->dim_features * this->dim_hidden, sizeof(float)); 

    // load weights of second layer
    filename = weight_dir.str().append("weight_2.bin");
    this->weight_2 = (float*)Model::read_binary(filename, this->dim_hidden * this->num_classes, sizeof(float));

    // load bias of first layer
    filename = weight_dir.str().append("bias_1.bin");
    this->bias_1 = (float*)Model::read_binary(filename, this->dim_hidden, sizeof(float));
    
    // load bias of second layer
    filename = weight_dir.str().append("bias_2.bin");
    this->bias_2 = (float*)Model::read_binary(filename, this->num_classes, sizeof(float));
}


void Model::free_model(void){
    free(this->weight_1);
    free(this->weight_2);
    free(this->bias_1);
    free(this->bias_2);
    free(this->labels);
    free(this->edges);
}


void Model::validate_input(std::string &dataset, int init_no, int seed){
    if (seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    if(seed < 0){
        std::cerr << "Seed has to be larger than 0!" << std::endl;
        exit(1);
    }

    std::cerr << "\tUsing dataset " << dataset << std::endl;
    std::cerr << "\tUsing init_no " << init_no << std::endl; 
    std::cerr << "\tUsing seed " << seed << std::endl << std::endl;
}


void Model::specify_problem(std::string &dataset, int* init_no, int* seed){
    std::cout << "READY" << std::endl;
    std::cout.flush();
    std::cerr << "Specify seed ";
    std::cin >> *seed;
    *init_no = -1; // -1, i.e. random weights
    dataset = "COAUTHORCS";

    validate_input(dataset, *init_no, *seed);
}


void Model::specify_problem(int argc, char** argv, std::string &dataset, int* init_no, int* seed){
    // validate arguments
    // you will receive a string specifying the dataset you work on and
    // a initialization number determining the initialization used in
    // training the network to get the correct weights
    if ((argc < 3) || (argc > 4)) {
        std::cerr << "Usage: " << argv[0] << "  DATASET  INITIALIZATION-NO [SEED]" << std::endl;
        exit(1);
    }

    std::cout << "READY" << std::endl;

    dataset = std::string(argv[1]);
    *init_no = std::stoi(argv[2]);

    if(argc == 3){
        *seed = 42;
    }
    else {
        *seed = std::stoi(argv[3]);
    }

    validate_input(dataset, *init_no, *seed);
}