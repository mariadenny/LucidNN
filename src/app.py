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

if 'hidden_layers' not in st.session_state:
    st.session_state.hidden_layers = [3] 
if 'network_data' not in st.session_state:
    st.session_state.network_data = {} 
if 'trained' not in st.session_state:
    st.session_state.trained = False

# --- HELPER FUNCTIONS ---
def get_topology(inputs, hidden, outputs):
    return [inputs] + hidden + [outputs]

def init_neuron_data(layer_idx, neuron_idx, num_prev_neurons):
    key = f"L{layer_idx}_N{neuron_idx}"
    if key not in st.session_state.network_data or \
       len(st.session_state.network_data[key]['weights']) != num_prev_neurons:
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-0.5, 0.5),
            "weights": [np.random.uniform(-1, 1) for _ in range(num_prev_neurons)]
        }
    return key

def calculate_stats(topology):
    total_layers = len(topology)
    total_neurons = sum(topology)
    total_connections = 0
    for i in range(len(topology) - 1):
        total_connections += topology[i] * topology[i+1]
    return total_layers, total_neurons, total_connections

# --- DIALOG: SET WEIGHTS & BIAS ---
@st.dialog("Set Weights & Bias")
def open_neuron_editor(layer_idx, neuron_idx, prev_layer_size):
    key = init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    data = st.session_state.network_data[key]

    st.subheader(f"Editing: Hidden Layer {layer_idx}, Neuron {neuron_idx+1}")
    
    # Bias
    new_bias = st.number_input("Bias", value=float(data['bias']), step=0.01, key=f"bias_{key}")
    
    st.markdown("---")
    st.markdown(f"**Weights (from previous layer: {prev_layer_size} inputs)**")
    
    # Weights
    new_weights = []
    cols = st.columns(3)
    for i in range(prev_layer_size):
        with cols[i % 3]:
            current_w_val = float(data['weights'][i])
            w = st.number_input(f"W_{i+1}", value=current_w_val, step=0.01, key=f"w_{key}_{i}")
            new_weights.append(w)
            
    if st.button("🎲 Randomize Values"):
        st.session_state.network_data[key]['bias'] = np.random.uniform(-1, 1)
        st.session_state.network_data[key]['weights'] = [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]
        st.rerun()

    if st.button("Save Changes", type="primary"):
        st.session_state.network_data[key]['bias'] = new_bias
        st.session_state.network_data[key]['weights'] = new_weights
        st.rerun()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Model Configuration")
    
    tab_arch, tab_hyper, tab_data = st.tabs(["Architecture", "Hyperparameters", "Data"])

    # --- TAB 1: ARCHITECTURE ---
    with tab_arch:
        input_nodes = st.number_input("Input Nodes", min_value=1, value=2, step=1)
        output_nodes = st.number_input("Output Nodes", min_value=1, value=1, step=1)
        
        st.markdown("---")
        st.subheader("Hidden Layers")
        
        if st.button("➕ Add Hidden Layer", use_container_width=True):
            st.session_state.layer_counter += 1
            st.session_state.layers.append({
                "id": st.session_state.layer_counter, 
                "neurons": 3
            })
            st.rerun()

        layers_to_remove = []
        for i, layer in enumerate(st.session_state.layers):
            st.markdown(f"**Layer {i+1}**")
            col1, col2 = st.columns([4, 1])
            with col1:
                new_val = st.number_input(
                    label="Neurons",
                    min_value=1,
                    value=layer['neurons'],
                    step=1,
                    key=f"layer_neurons_{layer['id']}",
                    label_visibility="collapsed"
                )
                st.session_state.layers[i]['neurons'] = new_val
            with col2:
                if st.button("✖", key=f"del_{layer['id']}", help="Delete this layer"):
                    layers_to_remove.append(i)

        if layers_to_remove:
            for index in sorted(layers_to_remove, reverse=True):
                del st.session_state.layers[index]
            st.rerun()

    # --- TAB 2: HYPERPARAMETERS ---
    with tab_hyper:
        st.subheader("Hyperparameters")
        activation = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Linear", "Leaky ReLU"])
        loss_fn = st.selectbox("Loss Function", ["Mean Squared Error (MSE)"])
        
        st.subheader("Training Config")
        epochs_setting = st.slider("Epochs", min_value=10, max_value=1000, step=10, value=100)
        learning_rate = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f")

    # --- TAB 3: TRAINING DATA ---
    with tab_data:
        st.subheader("Training Sample")
        st.caption("Provide the data you want the model to train on.")
        
        st.markdown("**Inputs (X)**")
        user_inputs = []
        for i in range(input_nodes):
            val = st.number_input(f"Input Feature {i+1}", value=1.0, step=0.1, key=f"in_{i}")
            user_inputs.append(val)
            
        st.markdown("---")
        st.markdown("**Expected Output (Y)**")
        user_targets = []
        for i in range(output_nodes):
            val = st.number_input(f"Target Output {i+1}", value=0.8, step=0.1, key=f"out_{i}")
            user_targets.append(val)

# --- COMPATIBILITY BRIDGE ---
st.session_state.hidden_layers = [layer["neurons"] for layer in st.session_state.layers]
activ_func = activation

# Recalculate topology based on new sidebar inputs
topology = get_topology(input_nodes, st.session_state.hidden_layers, output_nodes)

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

    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', splines='line', bgcolor='transparent')
    
    for l_idx, count in enumerate(topology):
        with graph.subgraph(name=f'cluster_{l_idx}') as c:
            c.attr(color='white', label=f'Layer {l_idx}')
            
            if l_idx == 0:
                color = '#FFCCCC' # Light Red
                label_prefix = 'x'
            elif l_idx == len(topology)-1:
                color = '#CCFFCC' # Light Green
                label_prefix = 'y'
            else:
                color = '#FFFFCC' # Light Yellow
                label_prefix = 'N'
            
            for n_idx in range(count):
                node_label = f"{label_prefix}{n_idx+1}"
                c.node(f'{l_idx}_{n_idx}', 
                       label=node_label, 
                       shape='circle', 
                       style='filled', 
                       fillcolor=color, 
                       color='black', 
                       fontcolor='black', 
                       width='0.6', 
                       fixedsize='true')

    for l_idx in range(len(topology) - 1):
        for n1 in range(topology[l_idx]):
            for n2 in range(topology[l_idx+1]):
                graph.edge(f'{l_idx}_{n1}', f'{l_idx+1}_{n2}', color='black')

    st.graphviz_chart(graph, use_container_width=True)

# --- RIGHT COLUMN: INTERACTION ---
with col_interact:
    st.subheader("Neuron Details")
    
    neuron_options = []
    for l in range(1, len(topology)): 
        layer_type = "Output" if l == len(topology)-1 else f"Hidden {l}"
        for n in range(topology[l]):
            neuron_options.append(f"Layer {l} ({layer_type}) - Neuron {n+1}")
            
    selected_neuron_str = st.selectbox("Select a Neuron to Inspect:", neuron_options)
    
    if selected_neuron_str:
        parts = selected_neuron_str.split(' ')
        l_idx = int(parts[1])
        n_idx = int(parts[-1]) - 1
        prev_layer_size = topology[l_idx - 1]
        
        key = init_neuron_data(l_idx, n_idx, prev_layer_size)
        curr_data = st.session_state.network_data[key]
        
        st.markdown(f"**Current Bias:** `{curr_data['bias']:.4f}`")
        
        with st.expander("View Weights", expanded=True):
            w_df = pd.DataFrame(curr_data['weights'], columns=["Weight Value"])
            if len(curr_data['weights']) == prev_layer_size:
                w_df.index = [f"Connection from Layer {l_idx-1} Neuron {i+1}" for i in range(prev_layer_size)]
            else:
                w_df.index = [f"Input {i+1}" for i in range(len(curr_data['weights']))]
            st.dataframe(w_df, use_container_width=True)

        if not st.session_state.trained:
            if st.button("🛠️ Edit Weights & Bias"):
                open_neuron_editor(l_idx, n_idx, prev_layer_size)
        
        else:
            st.info(f"Average Weight Over {epochs_setting} Epochs")
            if key in st.session_state.training_history:
                history_data = st.session_state.training_history[key]
                avg_weights = [np.mean(epoch_weights) for epoch_weights in history_data]
                chart_data = pd.DataFrame({"Epoch": range(len(avg_weights)), "Avg Weight": avg_weights})
                st.line_chart(chart_data, x="Epoch", y="Avg Weight", height=250)

# --- TRAIN BUTTON ---
st.markdown("---")
if st.button("🚀 Train Model (End-to-End)", type="primary", use_container_width=True):
    
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
        st.info("Next step: Run the C++ executable with this config file.")
    except Exception as e:
        st.error(f"Failed to generate config.json: {e}")
