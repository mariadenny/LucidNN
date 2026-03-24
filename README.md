# LucidNN

A neural network trainer and visualizer with a C++ backend and a Streamlit frontend.

## Project Structure

```
LucidNN/
├── .devcontainer
    ├── devcontainer.json
└── src/
    ├── app.py
    ├── main.cpp
    ├── Network.cpp / .h
    ├── Layer.cpp / .h
    ├── Neuron.cpp / .h
    ├── Activation.cpp / .h
    ├── JsonHandler.cpp / .h
    └── nlohmann/
        └── json.hpp
├── .gitignore
├── CMakeLists.txt
├── packages.txt
├── requirements.txt


```

## Requirements

- C++17 compiler (g++ or clang++)
- CMake 3.10+
- Python 3.x
- pip packages: `streamlit`, `pandas`, `numpy`, `plotly`

## Build the Backend

Run from the `LucidNN/` root:

```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

## Run the Frontend

```bash
streamlit run src/app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Configure your network architecture and hyperparameters in the sidebar
2. Click **Train Model** — the C++ backend trains and saves `results.json` and `model.json`
3. Load results to explore training history, loss curves, and matrix math
4. Use the **Predictions** tab to run inference on your trained model
