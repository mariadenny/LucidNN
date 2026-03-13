#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>

// The Labels (Tags)
enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    LEAKY_RELU
};

// The Math (Forward & Derivative)
class Activation {
public:
    static double activate(double x, ActivationType type);
    static double derivative(double x, ActivationType type);
    static std::string toString(ActivationType type);
    static ActivationType fromString(const std::string& s);
};

#endif