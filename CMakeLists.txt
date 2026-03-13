cmake_minimum_required(VERSION 3.10)

# Name of your project
project(LucidNNBackend)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Create the executable named 'app' from your source files
add_executable(app 
    main.cpp
    Activation.cpp
    JsonHandler.cpp
    Layer.cpp
    Network.cpp
    Neuron.cpp
)

# Tell the compiler to look for header files in the current source directory
# This allows it to find the local "nlohmann" folder
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})