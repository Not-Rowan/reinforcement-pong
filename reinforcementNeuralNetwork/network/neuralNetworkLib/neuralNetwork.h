// define neuralNetwork.h and include only once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// By: Rowan Rothe

// Declare network structures

struct Layer {
    float **incomingWeights;
    float *biases;
    float *values;
    float *gradients;
    int neuronCount;
};
typedef struct Layer Layer;

struct Network {
    Layer *layers;
    int layerCount;
};
typedef struct Network Network;

// Redeclare network functions
Network *createNetwork(int inputNodes, int hiddenLayers, int *hiddenNodes, int outputNodes);
void freeNetwork(Network *network);
// activation codes: 0 = sigmoid, 1 = relu, 2 = softmax (only for output activation)
void feedForward(Network *network, float *input, int hiddenActivation, int outputActivation);
void printNetwork(Network *network);
// activation codes: 0 = sigmoid, 1 = relu, 2 = tanh, 3 = linear, 4 = softmax (only for output activation)
void backPropagate(Network *network, float *expectedOutputs, float learningRate, int hiddenActivation, int outputActivation);
void reinforceNetwork(Network *network, float *expectedOutputs, float learningRate, int hiddenActivation, int outputActivation);
void copyNetwork(Network *destination, Network *source);
void copyNetworkGradually(Network *destination, Network *source, float rate);
void exportNetworkJSON(Network *network, char *filename);
Network *importNetworkJSON(char *filename);

#endif
