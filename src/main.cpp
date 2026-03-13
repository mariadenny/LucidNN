#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <exception>

#include <nlohmann/json.hpp>

#include "Network.h"
#include "JsonHandler.h"
int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << "Usage: ./app config.json\n";
        return 1;
    }

    try {
        std::ifstream in(argv[1]);
        nlohmann::json request;
        in >> request;

        std::string type = request["type"];

        // ===========================
        // TRAIN MODE
        // ===========================
        if (type == "INIT_NETWORK") {

            NetworkConfig config =
                JsonHandler::parseInitNetwork(argv[1]);

            Network net;
            net.initNetwork(config.input_size,
                            config.hidden_layers,
                            config.output_layer);

            auto results = net.trainAndReturnHistory(
                config.training_data.inputs,
                config.training_data.targets,
                config.epochs,
                config.learning_rate
            );

            // Save full training history
            std::ofstream out("results.json");
            out << results.dump(4);

            // Save final trained model
            net.saveModel("model.json");

            std::cout << "Training complete. Model saved.\n";
        }

        // ===========================
        // PREDICT MODE
        // ===========================
        else if (type == "PREDICT") {

            Network net;

            // Fully restore model (architecture + weights)
            net.loadFullModel("model.json");

            std::vector<double> input =
                request["input"].get<std::vector<double>>();

            std::vector<double> output =
                net.forward(input);

            nlohmann::json result;
            result["status"] = "success";
            result["prediction"] = output;

            std::ofstream out("prediction.json");
            out << result.dump(4);
        }

        else {
            std::cerr << "Unknown request type\n";
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}