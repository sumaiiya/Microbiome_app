import streamlit as st
import importlib.util
import os
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tempfile
#importlib.util
# Load simulation module from file
module_path = "kombucha_simulation.py"

#module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'kombucha_simulation.py'))


@st.cache_resource
def load_simulation_module():
    """Load the simulation module"""
    if not os.path.exists(module_path):
        st.error(f"Simulation module not found at {module_path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("kombucha_simulation", module_path)
        if spec is None:
            st.error(f"Failed to create spec for {module_path}")
            return None
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        return sim
    except Exception as e:
        st.error(f"Error loading simulation module: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title=" Microbial Simulation",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍶 Microbial Simulation")
st.markdown("Interactive simulation of microbial dynamics ")

# Load simulation module
sim = load_simulation_module()
if sim is None:
    st.stop()

# Check if database classes are available
if not getattr(sim, "ACTUAL_DB_AVAILABLE", False):
    st.error("❌ Database classes not available!")
    st.markdown("""
    **Setup Requirements:**
    1. Clone the simulation_envs repository
    2. Update the paths in `kombucha_simulation.py`:
       - `DB_SCRIPTS_PATH`
       - `CORE_SCRIPTS_PATH` 
       - `FILES_PATH`
       - `DATABASEFOLDER`
    3. Ensure the database file exists and is populated
    """)
    st.stop()


if sim.DB is None:
    st.error("❌ Database not found!")
    st.markdown(f"""
    **Database Setup:**
    - Looking for database at: `{os.path.join(sim.DATABASEFOLDER, sim.DATABASENAME)}`
    - Please ensure the database file exists and paths are correct
    """)
    st.stop()

st.success("✅ Database and classes loaded successfully!")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Extract current parameter values
current_params = {
    "max_steps": getattr(sim, 'param_max_steps', 240),
    "reward_scale": getattr(sim, 'param_reward_scale', 100),
    "simul_time": getattr(sim, 'param_simul_time', 1),
    "simul_steps": getattr(sim, 'param_simul_steps', 100),
    "dilution": getattr(sim, 'param_dilution', 0.5),
    "volume": getattr(sim, 'param_volume', 100),
    "max_dilution": getattr(sim, 'param_max_dilution', 60),
    "unit_change": getattr(sim, 'param_unit_change', 0.01),
    "target_ethanol": getattr(sim, 'param_target_ethanol', 10.0)
}

# Parameter inputs
user_params = {}

st.sidebar.subheader("Environment Parameters")
user_params['param_max_steps'] = st.sidebar.slider(
    "Max Steps", 
    min_value=10, 
    max_value=500, 
    value=current_params['max_steps'],
    help="Maximum number of simulation steps"
)

user_params['param_dilution'] = st.sidebar.slider(
    "Initial Dilution Rate", 
    min_value=0.0, 
    max_value=2.0, 
    value=current_params['dilution'],
    step=0.1,
    help="Initial dilution rate for the fermentation"
)

user_params['param_volume'] = st.sidebar.slider(
    "Reactor Volume (L)", 
    min_value=50, 
    max_value=500, 
    value=current_params['volume'],
    help="Volume of the fermentation reactor"
)

st.sidebar.subheader("Target Parameters")
user_params['param_target_ethanol'] = st.sidebar.slider(
    "Target Ethanol (g/L)", 
    min_value=5.0, 
    max_value=20.0, 
    value=current_params['target_ethanol'],
    step=0.5,
    help="Target ethanol concentration for optimization"
)

st.sidebar.subheader("Advanced Parameters")
with st.sidebar.expander("Advanced Settings"):
    user_params['param_max_dilution'] = st.slider(
        "Max Dilution Rate", 
        min_value=10, 
        max_value=100, 
        value=current_params['max_dilution'],
        help="Maximum allowed dilution rate"
    )
    
    user_params['param_unit_change'] = st.slider(
        "Unit Change", 
        min_value=0.001, 
        max_value=0.1, 
        value=current_params['unit_change'],
        step=0.001,
        format="%.3f",
        help="Step size for dilution rate changes"
    )

# Simulation type selection
st.header("Simulation Configuration")
sim_type = st.selectbox(
    "Choose simulation type", 
    ["Random Policy", "Environment Test", "State Analysis"],
    help="Select the type of simulation to run"
)

# Additional options
col1, col2 = st.columns(2)
with col1:
    show_debug = st.checkbox("Show Debug Information", value=False)
with col2:
    show_detailed_plots = st.checkbox("Show Detailed Plots", value=True)

# Run simulation button
#if st.button("🚀 Run Simulation", type="primary"):
if "simulation_ran" not in st.session_state:
    st.session_state.simulation_ran = False

if st.button("🚀 Run Simulation", type="primary"):
    st.session_state.simulation_ran = True

if st.session_state.simulation_ran:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Plots", "🔍 Analysis", "🐛 Debug"])
    
    # Create tabs for different outputs
    #tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Plots", "🔍 Analysis", "🐛 Debug"])
    
    with st.spinner("Running simulation..."):
        try:
            # Update simulation parameters
            for key, value in user_params.items():
                if hasattr(sim, key):
                    setattr(sim, key, value)
            
            # Run simulation
            if sim_type == "Environment Test":
                env = sim.KombuchaGym(max_steps=min(user_params['param_max_steps'], 50))
                obs, _ = env.reset()
                
                with tab1:
                    st.success("Environment created successfully!")
                    st.write(f"**Observation Space:** {env.observation_space}")
                    st.write(f"**Action Space:** {env.action_space}")
                    st.write(f"**Initial State Shape:** {obs.shape}")
                    st.write(f"**Initial State Values:** {obs}")
                
                with tab4:
                    if show_debug:
                        st.subheader("Debug Information")
                        st.write("**Environment Details:**")
                        st.json({
                            "current_step": env.current_step,
                            "dilution": env.dilution,
                            "max_steps": env.max_steps,
                            "volume": env.volume
                        })
                        st.write("**Reactor State:**")
                        st.write(f"Metabolites: {env.reactor.metabolome.metabolites}")
                        st.write(f"Concentrations: {env.reactor.metabolome.concentration}")
                        st.write(f"Bacterial populations: {[(k, v.count) for k, v in env.reactor.microbiome.subpopD.items()]}")
            
            elif sim_type == "State Analysis":
                env = sim.KombuchaGym(max_steps=20)
                states_data = []
                obs, _ = env.reset()
                states_data.append(obs.copy())
                
                for i in range(10):
                    action = env.action_space.sample()
                    obs, reward, done, _, info = env.step(action)
                    states_data.append(obs.copy())
                    if done:
                        break
                
                states_df = pd.DataFrame(states_data)
                states_df.columns = [f"State_{i}" for i in range(states_df.shape[1])]
                states_df['Step'] = range(len(states_df))
                
                with tab1:
                    st.subheader("State Analysis Results")
                    st.write(f"**Number of states:** {states_df.shape[1] - 1}")
                    st.write(f"**Steps analyzed:** {len(states_df)}")
                    st.subheader("State Statistics")
                    st.dataframe(states_df.describe())

                with tab2:
                    st.subheader("State Evolution")
                    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                    axes = axes.flatten()
                    for i in range(min(12, states_df.shape[1] - 1)):
                        col_name = f"State_{i}"
                        axes[i].plot(states_df['Step'], states_df[col_name], 'o-')
                        axes[i].set_title(f'{col_name}')
                        axes[i].set_xlabel('Step')
                        axes[i].grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
            

                
                with tab3:
                    st.subheader("Detailed State Analysis")
                    state_cols = [col for col in states_df.columns if col.startswith('State_')]
                    corr_matrix = states_df[state_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(state_cols)))
                    ax.set_yticks(range(len(state_cols)))
                    ax.set_xticklabels(state_cols, rotation=45)
                    ax.set_yticklabels(state_cols)
                    plt.colorbar(im)
                    plt.title('State Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            else:  # Random Policy
                result = sim.main(**user_params)
                #st.write("Simulation Output:", result)
                if result:
                    df = pd.DataFrame(result)
                    with tab1:
                        st.subheader("Simulation Results")
                        st.dataframe(df)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Final pH", f"{df['pH'].iloc[-1]:.2f}")
                        with col2:
                            st.metric("Final Ethanol", f"{df['ethanol'].iloc[-1]:.2f} g/L")
                        with col3:
                            st.metric("Avg Reward", f"{df['reward'].mean():.4f}")
                        with col4:
                            st.metric("Total Steps", len(df))
                    with tab2:
                        st.subheader("Simulation Time Series Plots")
                        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

                        axes[0].plot(df.index, df['pH'], label='pH', color='blue')
                        axes[0].set_ylabel('pH')
                        axes[0].grid(True)
                        axes[0].legend()

                        axes[1].plot(df.index, df['ethanol'], label='Ethanol (g/L)', color='green')
                        axes[1].set_ylabel('Ethanol (g/L)')
                        axes[1].grid(True)
                        axes[1].legend()

                        axes[2].plot(df.index, df['reward'], label='Reward', color='red')
                        axes[2].set_ylabel('Reward')
                        axes[2].set_xlabel('Step')
                        axes[2].grid(True)
                        axes[2].legend()

                        plt.tight_layout()
                        st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
