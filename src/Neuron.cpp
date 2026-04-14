#include "Neuron.h"
#include <cstdlib>

Neuron::Neuron(int num_inputs) :
bias(0.0), activation(0.0), z_value(0.0), delta(0.0), grad_bias(0.0) { 
    weights.resize(num_inputs);
    grad_weights.resize(num_inputs, 0.0);
    for (int i = 0; i < num_inputs; ++i) {
        weights[i] = ((double) rand() / RAND_MAX) - 0.5; 
    }
}