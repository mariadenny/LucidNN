import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LucidNN", layout="wide", page_icon="🧠")

# --- TITLE SECTION ---
st.title("LucidNN 🧠")
st.caption("Interactive Neural Network Designer & Neuron Editor")
st.markdown("---")

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
    """Derives the simple list of node counts directly from the complex dictionary."""
    hidden = [layer["neurons"] for layer in layers_dict]
    return [inputs] + hidden + [outputs]

def init_neuron_data(layer_idx, neuron_idx, num_prev_neurons):
    """Initializes random weights and biases if they don't exist yet."""
    key = f"L{layer_idx}_N{neuron_idx}"
    if key not in st.session_state.network_data or len(st.session_state.network_data[key]['weights']) != num_prev_neurons:
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-0.5, 0.5),
            "weights": [np.random.uniform(-1, 1) for _ in range(num_prev_neurons)]
        }
    return key

def initialize_all_neurons(topology):
    """Pre-fills the entire network state with random weights so config.json is complete."""
    for l_idx in range(1, len(topology)):
        prev_layer_size = topology[l_idx - 1]
        for n_idx in range(topology[l_idx]):
            init_neuron_data(l_idx, n_idx, prev_layer_size)

def calculate_stats(topology):
    total_layers = len(topology)
    total_neurons = sum(topology)
    total_connections = sum(topology[i] * topology[i+1] for i in range(len(topology) - 1))
    return total_layers, total_neurons, total_connections

def draw_network_graph(topology):
    """Encapsulates the verbose Graphviz drawing logic to keep the main layout clean."""
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', splines='line', bgcolor='transparent')
    
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
                # FIXED: Removed the +1 so nodes start at x0, N0, y0
                node_label = f"{label_prefix}{n_idx}" 
                c.node(f'{l_idx}_{n_idx}', label=node_label, shape='circle', style='filled', 
                       fillcolor=color, color='black', fontcolor='black', width='0.6', fixedsize='true')

    for l_idx in range(len(topology) - 1):
        for n1 in range(topology[l_idx]):
            for n2 in range(topology[l_idx+1]):
                graph.edge(f'{l_idx}_{n1}', f'{l_idx+1}_{n2}', color='black')
                
    return graph

# --- DIALOG: SET WEIGHTS & BIAS ---
@st.dialog("Set Weights & Bias")
def open_neuron_editor(layer_idx, neuron_idx, prev_layer_size):
    key = init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    data = st.session_state.network_data[key]

    st.subheader(f"Editing: Layer {layer_idx}, Neuron {neuron_idx}")
    
    # UPDATED: Added min_value and max_value constraints for Bias
    safe_bias = max(-10.0, min(10.0, float(data['bias'])))
    new_bias = st.number_input("Bias", min_value=-10.0, max_value=10.0, value=safe_bias, step=0.01, key=f"bias_{key}")
    
    st.markdown("---")
    st.markdown(f"**Weights (from previous layer: {prev_layer_size} nodes)**")
    
    new_weights = []
    cols = st.columns(3)
    for i in range(prev_layer_size):
        with cols[i % 3]:
            # UPDATED: Added min_value and max_value constraints for each Weight
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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Model Configuration")
    tab_arch, tab_hyper, tab_data = st.tabs(["Architecture", "Hyperparameters", "Data"])

    with tab_arch:
        input_nodes = st.number_input("Input Nodes", min_value=1, max_value=5, value=2, step=1)
        output_nodes = st.number_input("Output Nodes", min_value=1, max_value=5, value=1, step=1)
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
            st.markdown(f"**Layer {i+1}**")
            col1, col2 = st.columns([4, 1])
            with col1:
                safe_value = min(layer['neurons'], 5) 
                st.session_state.layers[i]['neurons'] = st.number_input(
                    label="Neurons", min_value=1, max_value=5, value=safe_value, step=1,
                    key=f"layer_neurons_{layer['id']}", label_visibility="collapsed"
                )
            with col2:
                if st.button("✖", key=f"del_{layer['id']}", help="Delete this layer"):
                    layers_to_remove.append(i)

        if layers_to_remove:
            for index in sorted(layers_to_remove, reverse=True):
                del st.session_state.layers[index]
            st.rerun()

    with tab_hyper:
        st.subheader("Hyperparameters")
        activation = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Linear", "Leaky ReLU"])
        loss_fn = st.selectbox("Loss Function", ["Mean Squared Error (MSE)"])
        st.subheader("Training Config")
        epochs_setting = st.slider("Epochs", min_value=10, max_value=1000, step=10, value=100)
        # UPDATED: Added min_value and max_value constraints
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0000, value=0.0100, step=0.001, format="%.4f")

    with tab_data:
        st.subheader("Training Sample")
        st.caption("Provide the data you want the model to train on.")
        st.markdown("**Inputs (X)**")
        user_inputs = [st.number_input(f"Input Feature {i}", value=1.0, step=0.1, key=f"in_{i}") for i in range(input_nodes)]
        st.markdown("---")
        st.markdown("**Expected Output (Y)**")
        user_targets = [st.number_input(f"Target Output {i}", value=0.8, step=0.1, key=f"out_{i}") for i in range(output_nodes)]

# Calculate topology for the main layout
topology = get_topology(input_nodes, st.session_state.layers, output_nodes)
activ_func = activation

# Force every single neuron to initialize immediately!
initialize_all_neurons(topology)

# --- MAIN PAGE LAYOUT ---
col_viz, col_interact = st.columns([3, 2])

# --- LEFT COLUMN: VISUALIZATION ---
with col_viz:
    st.subheader("Network Architecture")
    t_layers, t_neurons, t_conns = calculate_stats(topology)
    s1, s2, s3 = st.columns(3)
    s1.metric("Layers", t_layers)
    s2.metric("Neurons", t_neurons)
    s3.metric("Connections", t_conns)
    st.graphviz_chart(draw_network_graph(topology), use_container_width=True)

# --- RIGHT COLUMN: INTERACTION ---
with col_interact:
    st.subheader("Neuron Details")
    neuron_options = [
        f"Layer {l} ({'Output' if l == len(topology)-1 else f'Hidden {l}'}) - Neuron {n}"
        for l in range(1, len(topology)) for n in range(topology[l])
    ]
            
    selected_neuron_str = st.selectbox("Select a Neuron to Inspect:", neuron_options)
    
    if selected_neuron_str:
        parts = selected_neuron_str.split(' ')
        l_idx, n_idx = int(parts[1]), int(parts[-1])
        prev_layer_size = topology[l_idx - 1]
        
        key = init_neuron_data(l_idx, n_idx, prev_layer_size) 
        curr_data = st.session_state.network_data[key]
        
        st.markdown(f"**Current Bias:** `{curr_data['bias']:.4f}`")
        
        with st.expander("View Weights", expanded=True):
            w_df = pd.DataFrame(curr_data['weights'], columns=["Weight Value"])
            
            w_df.index = [f"Connection from Layer {l_idx-1} Neuron {i}" for i in range(prev_layer_size)] if len(curr_data['weights']) == prev_layer_size else [f"Input {i}" for i in range(len(curr_data['weights']))]
            st.dataframe(w_df, use_container_width=True)

        if st.button("🛠️ Edit Weights & Bias"):
            open_neuron_editor(l_idx, n_idx, prev_layer_size)

# --- TRAIN BUTTON ---
st.markdown("---")
if st.button("🚀 Train Model", type="primary", use_container_width=True):
    
    # 1. Prepare the Network Architecture block
    hidden_layers_config = []
    for layer in st.session_state.layers:
        hidden_layers_config.append({
            "neurons": layer["neurons"],
            "activation": activ_func.lower()
        })
        
    # 2. Construct the full JSON payload
    config_data = {
        "type": "INIT_NETWORK",
        "network": {
            "input_size": input_nodes,
            "hidden_layers": hidden_layers_config,
            "output_layer": {
                "neurons": output_nodes,
                "activation": activ_func.lower()
            }
        },
        "hyperparameters": {
            "epochs": epochs_setting,
            "learning_rate": learning_rate
        },
        "training_data": {
            "inputs": user_inputs,
            "targets": user_targets
        },
        "initial_state": st.session_state.network_data
    }
    
    # 3. Write to config.json
    try:
        with open("config.json", "w") as f:
            json.dump(config_data, f, indent=4)
        st.success("Configuration Ready! `config.json` generated successfully.")
        st.info("Next step: Run your C++ executable in the terminal (`./app config.json`).")
        
        # --- ADDED: Save config to session state so it stays on screen ---
        st.session_state.last_config = config_data
        st.session_state.ready_for_results = True
        
    except Exception as e:
        st.error(f"Failed to generate config.json: {e}")

# --- RESULTS SECTION ---
if st.session_state.get("ready_for_results", False):
    st.markdown("---")
    st.subheader("📈 Training Results")

    # 1. The Load Button (Saves data to session state so it doesn't disappear)
    if st.button("📊 Load Results", use_container_width=True):
        try:
            with open("results.json", "r") as f:
                st.session_state.results_data = json.load(f)
            
            if st.session_state.results_data.get("status") == "success":
                st.session_state.results_loaded = True
            else:
                st.error("Training did not complete successfully.")
        except FileNotFoundError:
            st.error("⚠️ `results.json` not found! Please run your C++ executable first.")
        except json.JSONDecodeError:
            st.error("⚠️ Failed to read `results.json`. The file might be corrupted.")

    # 2. The Interactive Dashboard (Only shows if data is loaded)
    if st.session_state.get("results_loaded", False):
        history = st.session_state.results_data.get("history", [])
        
        if not history:
            st.warning("Training history is empty.")
        else:
            st.success("Results loaded successfully!")
            
            # --- OVERALL LOSS CURVE ---
            epochs = [step["epoch"] for step in history]
            errors = [step["error"] for step in history]
            
            st.markdown("#### Loss Curve (Mean Squared Error)")
            chart_data = pd.DataFrame({"Epoch": epochs, "Error": errors})
            st.line_chart(chart_data, x="Epoch", y="Error", height=300)
            
            # --- 🕵️ EPOCH INSPECTOR (THE SLIDER) ---
            st.markdown("---")
            st.markdown("### 🕵️ Epoch Inspector")
            
            max_epoch = len(history)
            
            # The interactive slider
            selected_epoch = st.slider(
                "Drag to inspect a specific epoch:", 
                min_value=1, 
                max_value=max_epoch, 
                value=max_epoch # Defaults to the final epoch
            )
            
            step_data = history[selected_epoch - 1] 
            
            # --- SNAPSHOT METRICS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epoch", step_data["epoch"])
                st.metric("Error (MSE)", f"{step_data['error']:.6f}")
            with col2:
                actual = [round(x, 4) for x in step_data["actual_output"]]
                st.write("**Network Prediction:**")
                st.code(str(actual))
            with col3:
                expected = [round(x, 4) for x in step_data["expected_output"]]
                st.write("**Expected Target:**")
                st.code(str(expected))
                
            # --- NETWORK STATE (WEIGHTS & BIASES) ---
            st.markdown(f"#### Internal State at Epoch {selected_epoch}")
            network_state = step_data["network_state"]
            
            # Create a clean grid layout for the neurons
            neuron_keys = list(network_state.keys())
            
            # Display 3 neurons per row so it doesn't take up too much vertical space
            cols = st.columns(3) 
            for i, neuron_key in enumerate(neuron_keys):
                params = network_state[neuron_key]
                col = cols[i % 3]
                
                with col:
                    with st.expander(f"Neuron: {neuron_key}", expanded=False):
                        st.markdown(f"**Bias:** `{params['bias']:.6f}`")
                        
                        # Format the weights into a clean vertical dataframe
                        weights = params["weights"]
                        w_df = pd.DataFrame(
                            [round(w, 6) for w in weights], 
                            columns=["Weight Value"], 
                            index=[f"W{w_idx}" for w_idx in range(len(weights))]
                        )
                        st.dataframe(w_df, use_container_width=True)
