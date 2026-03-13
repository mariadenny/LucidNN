#ifndef JSON_HANDLER_H
#define JSON_HANDLER_H

#include <string>
#include <vector>
#include "Layer.h"
#include <utility>
#include <nlohmann/json.hpp>

struct TrainingData {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
};

struct NetworkConfig {
    int input_size;
    std::vector<std::pair<int, ActivationType>> hidden_layers;
    std::pair<int, ActivationType> output_layer;

    int epochs;
    double learning_rate;

    TrainingData training_data;

    nlohmann::json initial_state;
};

class JsonHandler {
public:
    static NetworkConfig parseInitNetwork(const std::string& file_path);
};

#endif