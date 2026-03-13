#include "Activation.h"
#include <cmath>
#include <stdexcept>
#include <string>

const double LEAKY_ALPHA = 0.01;

// --- FORWARD PASS MATH ---
double Activation::activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + std::exp(-x));
            
        case ActivationType::RELU:
            return (x > 0.0) ? x : 0.0;

        case ActivationType::LEAKY_RELU: 
            return (x > 0.0) ? x : LEAKY_ALPHA * x;
            
        case ActivationType::TANH:
            return std::tanh(x);
            
        case ActivationType::LINEAR:
            return x;
            
        default:
            throw std::runtime_error("Unknown activation type during forward pass.");
    }
}

// --- BACKWARD PASS MATH (DERIVATIVES) ---
double Activation::derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: {
            double sig = activate(x, ActivationType::SIGMOID);
            return sig * (1.0 - sig);
        }
        case ActivationType::RELU:
            return (x > 0.0) ? 1.0 : 0.0;

        case ActivationType::LEAKY_RELU: 
            return (x > 0.0) ? 1.0 : LEAKY_ALPHA;
            
        case ActivationType::TANH: {
            double t = std::tanh(x);
            return 1.0 - (t * t);
        }
        case ActivationType::LINEAR:
            return 1.0;
            
        default:
            throw std::runtime_error("Unknown activation type during backward pass.");
    }
}

// --- ENUM → STRING ---
std::string Activation::toString(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "sigmoid";
        case ActivationType::RELU: return "relu";
        case ActivationType::LEAKY_RELU: return "leaky relu";
        case ActivationType::TANH: return "tanh";
        case ActivationType::LINEAR: return "linear";
        default:
            throw std::runtime_error("Unknown activation type.");
    }
}

// --- STRING → ENUM ---
ActivationType Activation::fromString(const std::string& s) {
    if (s == "sigmoid") return ActivationType::SIGMOID;
    if (s == "relu") return ActivationType::RELU;
    if (s == "leaky relu") return ActivationType::LEAKY_RELU;
    if (s == "tanh") return ActivationType::TANH;
    if (s == "linear") return ActivationType::LINEAR;

    throw std::runtime_error("Unknown activation string: " + s);
}