# Data
Before being able to use datasets, you have to extract them from the `.zip` archives. 

- each dataset will be placed in a directory named accordingly, i.e. `./DATASET/`, e.g. `./CORA/`
- in those directories, you will find some basic information in the file `meta.txt` for each dataset, ground-truth labels in `y.bin` and the definition of edges in `edges.bin`
- In `DATASET/X` you will find the features of the nodes for each node separately, e.g. `DATASET/X/x_1.bin` will contain the input features for the node having ID 1
- `./DATASET/INIT_NO/`, e.g. `./Cora/1/` will contain the weights and biases of the model for different random initializations of the model's weights when training them, i.e. `weight_1.bin`, `weight_2.bin`, `bias_1.bin`, and `bias_2.bin`

# Testing
- for testing, we will use randomly initialized weights for the COAUTHOR dataset
- for testing your code, you might find those actual datasets with reasonable accuracy values more accuracte
- also note that those datasets have different characteristics and might help you identifying runtime characteristics

