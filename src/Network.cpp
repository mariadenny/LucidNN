#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <fstream> 

void Network::initNetwork(
    int input_size,
    const std::vector<std::pair<int, ActivationType>>& hidden_layers,
    std::pair<int, ActivationType> output_layer)
{
    layers.clear();

    if (input_size <= 0)
        throw std::runtime_error("Input size must be > 0");

    // Input layer (no weights)
    layers.emplace_back(input_size, 0, ActivationType::LINEAR);

    int previous_size = input_size;

    // Hidden layers
    for (const auto& hl : hidden_layers) {
        int neurons = hl.first;
        ActivationType type = hl.second;

        if (neurons <= 0)
            throw std::runtime_error("Hidden layer must have > 0 neurons");

        layers.emplace_back(neurons, previous_size, type);
        previous_size = neurons;
    }

    // Output layer
    int output_neurons = output_layer.first;
    ActivationType output_type = output_layer.second;

    if (output_neurons <= 0)
        throw std::runtime_error("Output layer must have > 0 neurons");

    layers.emplace_back(output_neurons, previous_size, output_type);
}

void Network::printStructure() const {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << " | Neurons: "
                  << layers[i].neurons.size();

        if (i > 0) {
            std::cout << " | Weights per neuron: "
                      << layers[i].neurons[0].weights.size();
        }

        std::cout << std::endl;
    }
}

void Network::setNeuronParams(int layer_idx, int neuron_idx, double bias, const std::vector<double>& weights) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    if (neuron_idx < 0 || neuron_idx >= layers[layer_idx].neurons.size()) return;
    layers[layer_idx].neurons[neuron_idx].bias = bias;
    layers[layer_idx].neurons[neuron_idx].weights = weights;
}

void Network::calculateNormalization(const std::vector<std::vector<double>>& inputs) {
    if (inputs.empty()) return;
    int features = inputs[0].size();
    input_mins.assign(features, std::numeric_limits<double>::max());
    input_maxs.assign(features, std::numeric_limits<double>::lowest());

    for (const auto& row : inputs) {
        for (size_t i = 0; i < features; ++i) {
            if (row[i] < input_mins[i]) input_mins[i] = row[i];
            if (row[i] > input_maxs[i]) input_maxs[i] = row[i];
        }
    }
}

std::vector<double> Network::normalize(const std::vector<double>& input) const {
    if (input_mins.empty() || input_maxs.empty() || input.size() != input_mins.size()) return input;
    std::vector<double> norm(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (input_maxs[i] == input_mins[i]) {
            norm[i] = input[i]; // avoid division by 0
        } else {
            norm[i] = (input[i] - input_mins[i]) / (input_maxs[i] - input_mins[i]);
        }
    }
    return norm;
}

std::vector<double> Network::forward(const std::vector<double>& input) {

    // Set input layer activations
    for (size_t i = 0; i < input.size(); ++i) {
        layers[0].neurons[i].activation = input[i];
    }

    // Forward through each layer
    for (size_t l = 1; l < layers.size(); ++l) {

        Layer& current = layers[l];
        Layer& previous = layers[l - 1];

        for (size_t n = 0; n < current.neurons.size(); ++n) {

            Neuron& neuron = current.neurons[n];
            double z = neuron.bias;

            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                z += neuron.weights[w] * previous.neurons[w].activation;
            }

            neuron.z_value = z;
            neuron.activation = Activation::activate(z, current.activationType);
        }
    }

    // Collect output layer activations
    std::vector<double> output;
    for (auto& neuron : layers.back().neurons) {
        output.push_back(neuron.activation);
    }

    return output;
}

double Network::computeMSE(const std::vector<double>& output,
                           const std::vector<double>& target) {
    double mse = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double diff = output[i] - target[i];
        mse += diff * diff;
    }
    return mse / output.size();
}

void Network::accumulateGradients(const std::vector<double>& target) {

    std::vector<std::vector<double>> deltas(layers.size());

    // --- Output layer ---
    size_t L = layers.size() - 1;
    deltas[L].resize(layers[L].neurons.size());

    for (size_t n = 0; n < layers[L].neurons.size(); ++n) {
        Neuron& neuron = layers[L].neurons[n];
        double error = neuron.activation - target[n];
        double delta = error * Activation::derivative(neuron.z_value, layers[L].activationType);
        
        deltas[L][n] = delta;
        neuron.delta = delta;
    }

    // --- Hidden layers ---
    for (int l = L - 1; l > 0; --l) {
        deltas[l].resize(layers[l].neurons.size());

        for (size_t i = 0; i < layers[l].neurons.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < layers[l + 1].neurons.size(); ++j) {
                sum += layers[l + 1].neurons[j].weights[i] * deltas[l + 1][j];
            }
            double delta = sum * Activation::derivative(layers[l].neurons[i].z_value, layers[l].activationType);
            
            deltas[l][i] = delta;
            layers[l].neurons[i].delta = delta; 
        }
    }

    // --- Accumulate gradients ---
    for (size_t l = 1; l < layers.size(); ++l) {
        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
            Neuron& neuron = layers[l].neurons[n];
            if (neuron.grad_weights.size() < neuron.weights.size()) {
                neuron.grad_weights.resize(neuron.weights.size(), 0.0);
            }
            neuron.grad_bias += deltas[l][n];

            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                neuron.grad_weights[w] += deltas[l][n] * layers[l - 1].neurons[w].activation;
            }
        }
    }
}

void Network::applyGradients(double learning_rate, int batch_size) {
    if (batch_size <= 0) batch_size = 1;
    for (size_t l = 1; l < layers.size(); ++l) {
        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
            Neuron& neuron = layers[l].neurons[n];
            neuron.bias -= learning_rate * (neuron.grad_bias / batch_size);
            neuron.grad_bias = 0.0;

            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                neuron.weights[w] -= learning_rate * (neuron.grad_weights[w] / batch_size);
                neuron.grad_weights[w] = 0.0;
            }
        }
    }
}

void Network::saveModel(const std::string& filename) {
    nlohmann::json model;

    model["network"]["input_size"] = layers[0].neurons.size();
    if (!input_mins.empty()) {
        model["network"]["input_mins"] = input_mins;
        model["network"]["input_maxs"] = input_maxs;
    }
    model["network"]["hidden_layers"] = nlohmann::json::array();

    for (size_t l = 1; l < layers.size() - 1; ++l) {
        nlohmann::json layer_info;
        layer_info["neurons"] = layers[l].neurons.size();
        layer_info["activation"] = Activation::toString(layers[l].activationType);
        model["network"]["hidden_layers"].push_back(layer_info);
    }

    size_t last = layers.size() - 1;
    model["network"]["output_layer"]["neurons"] = layers[last].neurons.size();
    model["network"]["output_layer"]["activation"] = Activation::toString(layers[last].activationType);

    for (size_t l = 1; l < layers.size(); ++l) {
        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
            std::string key = "L" + std::to_string(l) + "_N" + std::to_string(n);
            model["weights"][key]["bias"] = layers[l].neurons[n].bias;
            model["weights"][key]["weights"] = layers[l].neurons[n].weights;
        }
    }

    std::ofstream out(filename);
    out << model.dump(4);
}

void Network::loadFullModel(const std::string& filename) {
    std::ifstream in(filename);
    nlohmann::json model;
    in >> model;

    int input_size = model["network"]["input_size"];
    if (model["network"].contains("input_mins")) {
        input_mins = model["network"]["input_mins"].get<std::vector<double>>();
        input_maxs = model["network"]["input_maxs"].get<std::vector<double>>();
    }
    
    std::vector<std::pair<int, ActivationType>> hidden_layers;

    for (auto& hl : model["network"]["hidden_layers"]) {
        int neurons = hl["neurons"];
        ActivationType act = Activation::fromString(hl["activation"]);
        hidden_layers.push_back({neurons, act});
    }

    int out_neurons = model["network"]["output_layer"]["neurons"];
    ActivationType out_act = Activation::fromString(model["network"]["output_layer"]["activation"]);

    initNetwork(input_size, hidden_layers, {out_neurons, out_act});

    for (size_t l = 1; l < layers.size(); ++l) {
        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
            std::string key = "L" + std::to_string(l) + "_N" + std::to_string(n);
            layers[l].neurons[n].bias = model["weights"][key]["bias"];
            layers[l].neurons[n].weights = model["weights"][key]["weights"].get<std::vector<double>>();
        }
    }
}

nlohmann::ordered_json Network::trainAndReturnHistory(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    int epochs,
    double learning_rate,
    int batch_size
) {
    nlohmann::ordered_json result;
    result["status"] = "success";
    result["history"] = nlohmann::json::array();

    if (inputs.size() != targets.size()) {
        throw std::runtime_error("Inputs and targets size mismatch");
    }
    
    calculateNormalization(inputs);
    
    if (batch_size <= 0) batch_size = 1;

    for (int e = 1; e <= epochs; ++e) {
        double total_error = 0.0;
        std::vector<double> last_output;
        int current_batch_count = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> norm_input = normalize(inputs[i]);
            std::vector<double> output = forward(norm_input);
            double error = computeMSE(output, targets[i]);
            total_error += error;
            
            accumulateGradients(targets[i]);
            current_batch_count++;
            
            if (current_batch_count == batch_size || i == inputs.size() - 1) {
                applyGradients(learning_rate, current_batch_count);
                current_batch_count = 0;
            }
            last_output = output; 
        }

        double avg_error = total_error / inputs.size();

        nlohmann::ordered_json snapshot;
        snapshot["epoch"] = e;
        snapshot["error"] = avg_error;
        snapshot["actual_output"] = last_output;
        snapshot["expected_output"] = targets.back();

        nlohmann::ordered_json network_state;
        for (size_t l = 1; l < layers.size(); ++l) {
            for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
                std::string key = "L" + std::to_string(l) + "_N" + std::to_string(n);
                network_state[key]["bias"] = layers[l].neurons[n].bias;
                network_state[key]["weights"] = layers[l].neurons[n].weights;
            }
        }
        snapshot["network_state"] = network_state;

        // --- NEW: Extracting Math Details for Matrices ---
        nlohmann::ordered_json math_details;
        
        // Save Input Layer Activations (A0)
        std::vector<std::vector<double>> A0_matrix;
        for (size_t i = 0; i < layers[0].neurons.size(); ++i) {
            A0_matrix.push_back({layers[0].neurons[i].activation});
        }
        math_details["Layer_0"]["A"] = A0_matrix;

        // Extract matrices for hidden and output layers
        for (size_t l = 1; l < layers.size(); ++l) {
            std::vector<std::vector<double>> W_matrix;
            std::vector<std::vector<double>> B_matrix;
            std::vector<std::vector<double>> Z_matrix;
            std::vector<std::vector<double>> A_matrix;
            std::vector<std::vector<double>> Delta_matrix;

            for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
                W_matrix.push_back(layers[l].neurons[n].weights);
                B_matrix.push_back({layers[l].neurons[n].bias});
                Z_matrix.push_back({layers[l].neurons[n].z_value});
                A_matrix.push_back({layers[l].neurons[n].activation});
                Delta_matrix.push_back({layers[l].neurons[n].delta});
            }

            math_details["Layer_" + std::to_string(l)]["W"] = W_matrix;
            math_details["Layer_" + std::to_string(l)]["B"] = B_matrix;
            math_details["Layer_" + std::to_string(l)]["Z"] = Z_matrix;
            math_details["Layer_" + std::to_string(l)]["A"] = A_matrix;
            math_details["Layer_" + std::to_string(l)]["Delta"] = Delta_matrix;
        }
        
        snapshot["math_details"] = math_details;
        result["history"].push_back(snapshot);
    }

    return result;
}