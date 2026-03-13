import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="LucidNN", layout="wide", page_icon="🧠")

# Force Sidebar to be 25% of Viewport Width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 25vw !important;
            max-width: 25vw !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- TITLE SECTION ---
st.title("LucidNN 🧠")
st.caption("Interactive Neural Network Designer & Educational Visualizer")

# --- SESSION STATE INITIALIZATION ---
if 'layers' not in st.session_state:
    st.session_state.layers = [{"id": 0, "neurons": 3}]
if 'layer_counter' not in st.session_state:
    st.session_state.layer_counter = 0
if 'network_data' not in st.session_state:
    st.session_state.network_data = {} 
if 'trained' not in st.session_state:
    st.session_state.trained = False

# --- FUNCTIONS ---
def get_topology(inputs, layers_dict, outputs):
    hidden = [layer["neurons"] for layer in layers_dict]
    return [inputs] + hidden + [outputs]

def init_neuron_data(layer_idx, neuron_idx, num_prev_neurons):
    key = f"L{layer_idx}_N{neuron_idx}"
    if key not in st.session_state.network_data or len(st.session_state.network_data[key]['weights']) != num_prev_neurons:
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-0.5, 0.5),
            "weights": [np.random.uniform(-1, 1) for _ in range(num_prev_neurons)]
        }
    return key

def initialize_all_neurons(topology):
    for l_idx in range(1, len(topology)):
        prev_layer_size = topology[l_idx - 1]
        for n_idx in range(topology[l_idx]):
            init_neuron_data(l_idx, n_idx, prev_layer_size)

def calculate_stats(topology):
    total_layers = len(topology)
    total_neurons = sum(topology)
    total_connections = sum(topology[i] * topology[i+1] for i in range(len(topology) - 1))
    return total_layers, total_neurons, total_connections

def to_latex_matrix(matrix_data):
    lines = []
    for row in matrix_data:
        lines.append(" & ".join([f"{val:.3f}" for val in row]))
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

def draw_network_graph(topology, network_state=None):
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', splines='line', bgcolor='transparent')
    
    max_abs_w = 0.001 
    if network_state:
        for key, data in network_state.items():
            for w in data.get("weights", []):
                if abs(w) > max_abs_w:
                    max_abs_w = abs(w)
    
    for l_idx, count in enumerate(topology):
        with graph.subgraph(name=f'cluster_{l_idx}') as c:
            c.attr(color='white', label=f'Layer {l_idx}')
            
            if l_idx == 0:
                color, label_prefix = '#FFCCCC', 'x' 
            elif l_idx == len(topology)-1:
                color, label_prefix = '#CCFFCC', 'y' 
            else:
                color, label_prefix = '#FFFFCC', 'N' 
            
            for n_idx in range(count):
                node_label = f"{label_prefix}{n_idx}" 
                c.node(f'{l_idx}_{n_idx}', label=node_label, shape='circle', style='filled', 
                       fillcolor=color, color='black', fontcolor='black', width='0.6', fixedsize='true')

    for l_idx in range(len(topology) - 1):
        for n1 in range(topology[l_idx]):
            for n2 in range(topology[l_idx+1]):
                edge_color = '#cccccc' 
                penwidth = '1.0'
                
                if network_state:
                    key = f"L{l_idx+1}_N{n2}"
                    if key in network_state:
                        weights = network_state[key].get("weights", [])
                        if n1 < len(weights):
                            w = weights[n1]
                            edge_color = '#2ca02c' if w > 0 else '#d62728' 
                            normalized_w = abs(w) / max_abs_w
                            thickness = 0.5 + (normalized_w * 4.5)
                            penwidth = str(round(thickness, 2)) 
                            
                graph.edge(f'{l_idx}_{n1}', f'{l_idx+1}_{n2}', color=edge_color, penwidth=penwidth)
                
    return graph

# --- DIALOG: SET WEIGHTS & BIAS ---
@st.dialog("Set Weights & Bias")
def open_neuron_editor(layer_idx, neuron_idx, prev_layer_size):
    key = init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    data = st.session_state.network_data[key]

    st.subheader(f"Editing: Layer {layer_idx}, Neuron {neuron_idx}")
    safe_bias = max(-10.0, min(10.0, float(data['bias'])))
    new_bias = st.number_input("Bias", min_value=-10.0, max_value=10.0, value=safe_bias, step=0.01, key=f"bias_{key}")
    
    st.markdown("---")
    st.markdown(f"**Weights (from previous layer: {prev_layer_size} nodes)**")
    
    new_weights = []
    cols = st.columns(3)
    for i in range(prev_layer_size):
        with cols[i % 3]:
            safe_weight = max(-10.0, min(10.0, float(data['weights'][i])))
            w = st.number_input(f"W_{i}", min_value=-10.0, max_value=10.0, value=safe_weight, step=0.01, key=f"w_{key}_{i}")
            new_weights.append(w)
            
    if st.button("🎲 Randomize Values"):
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-1, 1),
            "weights": [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]
        }
        st.rerun()

    if st.button("Save Changes", type="primary"):
        st.session_state.network_data[key]['bias'] = new_bias
        st.session_state.network_data[key]['weights'] = new_weights
        st.rerun()

# --- SIDEBAR CONFIGURATION (WITH TOOLTIPS) ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    tab_arch, tab_hyper, tab_data = st.tabs(["Architecture", "Hyperparameters", "Data"])

    with tab_arch:
        input_nodes = st.number_input("Input Nodes", min_value=1, max_value=5, value=2, step=1, help="The number of features passed into the network.")
        output_nodes = st.number_input("Output Nodes", min_value=1, max_value=5, value=1, step=1, help="The number of final predictions the network makes.")
        st.markdown("---")
        st.subheader("Hidden Layers")
        max_layers_reached = len(st.session_state.layers) >= 5
        if st.button("➕ Add Hidden Layer", use_container_width=True, disabled=max_layers_reached):
            st.session_state.layer_counter += 1
            st.session_state.layers.append({"id": st.session_state.layer_counter, "neurons": 3})
            st.rerun()
        if max_layers_reached:
            st.caption("⚠️ Maximum of 5 hidden layers reached.")
        
        layers_to_remove = []
        for i, layer in enumerate(st.session_state.layers):
            col1, col2 = st.columns([4, 1])
            with col1:
                safe_value = min(layer['neurons'], 5) 
                st.session_state.layers[i]['neurons'] = st.number_input(
                    label=f"Layer {i+1} Neurons", min_value=1, max_value=5, value=safe_value, step=1,
                    key=f"layer_neurons_{layer['id']}"
                )
            with col2:
                st.write("") # Spacing
                if st.button("✖", key=f"del_{layer['id']}", help="Delete this layer"):
                    layers_to_remove.append(i)

        if layers_to_remove:
            for index in sorted(layers_to_remove, reverse=True):
                del st.session_state.layers[index]
            st.rerun()

    with tab_hyper:
        st.subheader("Hyperparameters")
        activation = st.selectbox(
            "Activation Function", 
            ["ReLU", "Sigmoid", "Tanh", "Linear", "Leaky ReLU"],
            help="The mathematical filter applied to a neuron's output. Introduces non-linearity to solve complex problems."
        )
        loss_fn = st.selectbox(
            "Loss Function", 
            ["Mean Squared Error (MSE)"],
            help="How the network measures its mistakes. MSE penalizes large errors heavily."
        )
        st.subheader("Training Config")
        epochs_setting = st.slider(
            "Epochs", min_value=10, max_value=5000, step=10, value=100,
            help="One epoch is one complete forward and backward pass of all your training data."
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=1.0000, value=0.0100, step=0.001, format="%.4f",
            help="The 'step size' the network takes when updating weights. Too high = erratic learning; Too low = very slow learning."
        )

    with tab_data:
        st.subheader("Training Dataset")
        st.caption("Add multiple rows to train on complex datasets (e.g., XOR gate).")
        
        x_cols = [f"Input X{i}" for i in range(input_nodes)]
        y_cols = [f"Target Y{i}" for i in range(output_nodes)]
        
        default_data = []
        if input_nodes == 2 and output_nodes == 1:
            default_data = [
                {x_cols[0]: 0.0, x_cols[1]: 0.0, y_cols[0]: 0.0},
                {x_cols[0]: 0.0, x_cols[1]: 1.0, y_cols[0]: 1.0},
                {x_cols[0]: 1.0, x_cols[1]: 0.0, y_cols[0]: 1.0},
                {x_cols[0]: 1.0, x_cols[1]: 1.0, y_cols[0]: 0.0},
            ]
        else:
            for _ in range(4):
                row = {col: 0.0 for col in x_cols + y_cols}
                default_data.append(row)
                
        df = pd.DataFrame(default_data)
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        user_inputs_2d = edited_df[x_cols].values.tolist()
        user_targets_2d = edited_df[y_cols].values.tolist()

topology = get_topology(input_nodes, st.session_state.layers, output_nodes)
activ_func = activation
initialize_all_neurons(topology)

# --- GLOBAL ACTION BUTTONS ---
col_act1, col_act2 = st.columns(2)

with col_act1:
    if st.button("🚀 1. Generate Config & Train", type="primary", use_container_width=True):
        if os.path.exists("results.json"): os.remove("results.json")
        
        hidden_layers_config = [{"neurons": l["neurons"], "activation": activ_func.lower()} for l in st.session_state.layers]
        valid_keys = [f"L{l}_N{n}" for l in range(1, len(topology)) for n in range(topology[l])]
        cleaned_network_state = {k: v for k, v in st.session_state.network_data.items() if k in valid_keys}

        config_data = {
            "type": "INIT_NETWORK",
            "network": {"input_size": input_nodes, "hidden_layers": hidden_layers_config, "output_layer": {"neurons": output_nodes, "activation": activ_func.lower()}},
            "hyperparameters": {"epochs": epochs_setting, "learning_rate": learning_rate},
            "training_data": {"inputs": user_inputs_2d, "targets": user_targets_2d},
            "initial_state": cleaned_network_state 
        }
        
        try:
            with open("config.json", "w") as f: json.dump(config_data, f, indent=4)
            st.success("Configuration Ready! Run `./build/app config.json` in your terminal.")
            st.session_state.ready_for_results = True
        except Exception as e:
            st.error(f"Failed to generate config.json: {e}")

with col_act2:
    if st.button("📊 2. Load C++ Results", use_container_width=True):
        try:
            with open("results.json", "r") as f:
                st.session_state.results_data = json.load(f)
            if st.session_state.results_data.get("status") == "success":
                st.session_state.results_loaded = True
                st.success("Results loaded successfully!")
            else:
                st.error("Training did not complete successfully.")
        except FileNotFoundError:
            st.error("⚠️ `results.json` not found! Please run your C++ executable first.")

# --- GLOBAL EPOCH SLIDER (Visible if results are loaded) ---
current_network_state = st.session_state.network_data 
step_data = None
history = []

if st.session_state.get("results_loaded", False):
    history = st.session_state.results_data.get("history", [])
    if history:
        st.markdown("### 🕵️ Global Epoch Inspector")
        max_epoch = len(history)
        selected_epoch = st.slider(
            "Scrub through time to see how the network evolved at this exact epoch across all tabs:", 
            min_value=1, max_value=max_epoch, value=max_epoch, key="epoch_slider"
        )
        selected_epoch = max(1, min(selected_epoch, max_epoch)) 
        step_data = history[selected_epoch - 1]
        current_network_state = step_data["network_state"]

st.markdown("---")

# --- THE MAIN TABBED INTERFACE ---
tab_diagram, tab_results, tab_math, tab_predict = st.tabs([
    "📊 Network Diagram", 
    "📈 Training Results", 
    "🧮 Matrix Math", 
    "🔮 Predictions"
])

# TAB 1: DIAGRAM
with tab_diagram:
    col_viz, col_interact = st.columns([3, 2])
    with col_viz:
        t_layers, t_neurons, t_conns = calculate_stats(topology)
        s1, s2, s3 = st.columns(3)
        s1.metric("Layers", t_layers)
        s2.metric("Neurons", t_neurons)
        s3.metric("Connections", t_conns)
        st.graphviz_chart(draw_network_graph(topology, current_network_state), use_container_width=True)

    with col_interact:
        st.subheader("Inspect/Edit Neurons")
        st.caption("Editing weights here applies to the INITIAL state before training.")
        neuron_options = [f"Layer {l} - Neuron {n}" for l in range(1, len(topology)) for n in range(topology[l])]
        selected_neuron_str = st.selectbox("Select a Neuron:", neuron_options)
        
        if selected_neuron_str:
            parts = selected_neuron_str.split(' ')
            l_idx, n_idx = int(parts[1]), int(parts[-1])
            prev_layer_size = topology[l_idx - 1]
            key = init_neuron_data(l_idx, n_idx, prev_layer_size) 
            
            # Show dynamic data if scrubbing, otherwise show initial state
            display_data = current_network_state[key] if key in current_network_state else st.session_state.network_data[key]
            
            st.markdown(f"**Current Bias:** `{display_data['bias']:.4f}`")
            with st.expander("View Weights", expanded=True):
                w_df = pd.DataFrame(display_data['weights'], columns=["Weight Value"])
                w_df.index = [f"from L{l_idx-1}_N{i}" for i in range(prev_layer_size)]
                st.dataframe(w_df, use_container_width=True)

            if st.button("🛠️ Edit Initial Weights & Bias"):
                open_neuron_editor(l_idx, n_idx, prev_layer_size)

# TAB 2: TRAINING RESULTS
with tab_results:
    if not st.session_state.get("results_loaded", False) or not history:
        st.info("Train the model and click 'Load C++ Results' to view graphs.")
    else:
        filtered_history = history[:selected_epoch]
        epochs = [step["epoch"] for step in filtered_history]
        errors = [step["error"] for step in filtered_history]
        
        st.markdown(f"#### Loss Curve up to Epoch {selected_epoch}")
        chart_data = pd.DataFrame({"Epoch": epochs, "Error": errors})
        fig = px.line(chart_data, x="Epoch", y="Error", markers=True)
        fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Epoch", step_data["epoch"])
            st.metric("Error (MSE)", f"{step_data['error']:.6f}")
        with col2:
            st.write("**Prediction (Last Sample):**")
            st.code(str([round(x, 4) for x in step_data["actual_output"]]))
        with col3:
            st.write("**Target (Last Sample):**")
            st.code(str([round(x, 4) for x in step_data["expected_output"]]))

# TAB 3: MATRIX MATH
with tab_math:
    if not st.session_state.get("results_loaded", False) or not step_data:
        st.info("Train the model and click 'Load C++ Results' to view math breakdowns.")
    else:
        st.markdown(f"### Mathematical Transformations at Epoch {selected_epoch}")
        st.caption("Visualizing the matrix math for the last processed sample of this epoch.")
        
        math_details = step_data.get("math_details", {})
        
        for l in range(1, len(topology)):
            st.markdown(f"##### 🔹 Layer {l} Math")
            layer_str = f"Layer_{l}"
            prev_layer_str = f"Layer_{l-1}"
            
            if layer_str in math_details and prev_layer_str in math_details:
                W = to_latex_matrix(math_details[layer_str]["W"])
                A_prev = to_latex_matrix(math_details[prev_layer_str]["A"])
                B = to_latex_matrix(math_details[layer_str]["B"])
                Z = to_latex_matrix(math_details[layer_str]["Z"])
                A = to_latex_matrix(math_details[layer_str]["A"])
                Delta = to_latex_matrix(math_details[layer_str]["Delta"])

                st.write("**1. Weighted Sum (Pre-activation)**")
                st.latex(rf"Z^{{({l})}} = W^{{({l})}} \cdot A^{{({l-1})}} + B^{{({l})}}")
                st.latex(rf"{Z} = {W} \cdot {A_prev} + {B}")
                
                st.write("**2. Activation Function**")
                st.latex(rf"A^{{({l})}} = \sigma(Z^{{({l})}}) = {A}")
                
                st.write("**3. Error Gradients (Deltas during Backprop)**")
                st.latex(rf"\delta^{{({l})}} = {Delta}")
                st.divider()

# TAB 4: PREDICTIONS
with tab_predict:
    st.markdown("### 🔮 Predict Using Trained Model")
    st.caption("Test your fully trained `model.json` by feeding it brand new, unseen data.")

    pred_cols = st.columns(input_nodes)
    pred_inputs = []
    for i in range(input_nodes):
        with pred_cols[i]:
            val = st.number_input(f"Input Feature {i}", value=0.0, step=0.1, key=f"pred_input_{i}")
            pred_inputs.append(val)

    col_req, col_res = st.columns(2)

    with col_req:
        if st.button("📝 1. Generate Prediction Request", use_container_width=True):
            if os.path.exists("prediction.json"): os.remove("prediction.json")
            if "prediction_result" in st.session_state: del st.session_state.prediction_result
                
            try:
                with open("predict_request.json", "w") as f:
                    json.dump({"type": "PREDICT", "input": pred_inputs}, f, indent=4)
                st.success("Request generated! Run `./build/app predict_request.json`.")
            except Exception as e:
                st.error(f"Failed to generate predict request: {e}")

    with col_res:
        if st.button("🔍 2. Load Prediction Result", type="primary", use_container_width=True):
            if os.path.exists("prediction.json"):
                try:
                    with open("prediction.json", "r") as f:
                        pred_res = json.load(f)
                    if pred_res.get("status") == "success":
                        st.session_state.prediction_result = pred_res
                    else:
                        st.error("Prediction failed or invalid format.")
                except json.JSONDecodeError:
                    st.error("⚠️ Corrupted `prediction.json`.")
            else:
                st.error("⚠️ `prediction.json` not found. Did you run the C++ executable?")

        if "prediction_result" in st.session_state:
            pred_res = st.session_state.prediction_result
            st.success("Prediction Loaded Successfully!")
            st.write("**Model Output:**")
            st.code(str([round(x, 6) for x in pred_res["prediction"]]))
            
            if st.button("🗑️ Clear Result"):
                del st.session_state.prediction_result
                if os.path.exists("prediction.json"): os.remove("prediction.json")
                st.rerun()