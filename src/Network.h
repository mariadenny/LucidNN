#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <nlohmann/json.hpp>
#include "Layer.h"

class Network {
public:
    std::vector<Layer> layers;

    void initNetwork(int input_size,
                     const std::vector<std::pair<int, ActivationType>>& hidden_layers,
                     std::pair<int, ActivationType> output_layer);

    void printStructure() const;

    std::vector<double> input_mins;
    std::vector<double> input_maxs;

    void calculateNormalization(const std::vector<std::vector<double>>& inputs);
    std::vector<double> normalize(const std::vector<double>& input) const;

    void setNeuronParams(int layer_idx, int neuron_idx, double bias, const std::vector<double>& weights);
    std::vector<double> forward(const std::vector<double>& input);
    double computeMSE(const std::vector<double>& output, const std::vector<double>& target);
    
    void accumulateGradients(const std::vector<double>& target);
    void applyGradients(double learning_rate, int batch_size);

    nlohmann::ordered_json trainAndReturnHistory(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        int epochs,
        double learning_rate,
        int batch_size
    );

    void saveModel(const std::string& filename);
    void loadFullModel(const std::string& filename);
};

#endif