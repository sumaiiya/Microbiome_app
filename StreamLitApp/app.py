import streamlit as st
import importlib.util
import os
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Load simulation module from file
module_path = "kombucha_simulation.py"

@st.cache_resource
def load_simulation_module():
    """Load the kombucha simulation module"""
    if not os.path.exists(module_path):
        st.error(f"Simulation module not found at {module_path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("kombucha_simulation", module_path)
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        return sim
    except Exception as e:
        st.error(f"Error loading simulation module: {e}")
        return None

# Page setup
st.set_page_config(page_title="Kombucha Simulation", page_icon="🍶", layout="wide")

st.title("🍶 Microbial Simulation Interface")
st.markdown("Interactive simulation of Kombucha microbial dynamics")

sim = load_simulation_module()
if sim is None:
    st.stop()

if not getattr(sim, "ACTUAL_DB_AVAILABLE", False):
    st.error("❌ Database classes not available!")
    st.stop()

if sim.DB is None:
    st.error("❌ Database not found!")
    st.stop()

st.success("✅ Database and classes loaded successfully!")

# Simulation type selection
sim_type = st.selectbox(
    "Choose simulation type", 
    ["Random Policy", "Environment Test", "State Analysis", "Direct Reactor ODE Simulation"],
    help="Select the type of simulation to run"
)

# Parameter sections
current_params = {
    "max_steps": getattr(sim, 'param_max_steps', 240),
    "reward_scale": getattr(sim, 'param_reward_scale', 100),
    "simul_time": getattr(sim, 'param_simul_time', 1),
    "simul_steps": getattr(sim, 'param_simul_steps', 100),
    "dilution": getattr(sim, 'param_dilution', 0.5),
    "volume": getattr(sim, 'param_volume', 100),  # fix: key is 'volume', not 'param_volume'
    "max_dilution": getattr(sim, 'param_max_dilution', 60),
    "unit_change": getattr(sim, 'param_unit_change', 0.01),
    "target_ethanol": getattr(sim, 'param_target_ethanol', 10.0)
}

# Sidebar parameters
user_params = {}

if sim_type != "Direct Reactor ODE Simulation":
    st.sidebar.header("Gym Simulation Parameters")

    user_params['param_max_steps'] = st.sidebar.slider("Max Steps", 10, 500, current_params['max_steps'])
    user_params['param_dilution'] = st.sidebar.slider("Dilution", 0.0, 2.0, current_params['dilution'], 0.1)
    user_params['param_volume'] = st.sidebar.slider("Volume (L)", 50, 500, current_params['volume'])  # fixed key here
    user_params['param_target_ethanol'] = st.sidebar.slider("Target Ethanol (g/L)", 5.0, 20.0, current_params['target_ethanol'], 0.5)

    with st.sidebar.expander("Advanced Settings"):
        user_params['param_max_dilution'] = st.slider("Max Dilution Rate", 10, 100, current_params['max_dilution'])
        user_params['param_unit_change'] = st.slider("Unit Change", 0.001, 0.1, current_params['unit_change'], 0.001, format="%.3f")
else:
    st.sidebar.header("ODE Reactor Setup")

    metabolome = sim.createMetabolome(sim.DB, mediaName='kombucha_media', pHFunc=sim.getpH)
    metabolome_feed = sim.createMetabolome(sim.DB, mediaName='kombucha_media', pHFunc=sim.getpH)

    microbiome = sim.Microbiome({
        'bb': sim.createBacteria(sim.DB, speciesID='bb', mediaName='kombucha_media'),
        'ki': sim.createBacteria(sim.DB, speciesID='ki', mediaName='kombucha_media')
    })

    microbiome_feed = sim.Microbiome({
        'bb': sim.createBacteria(sim.DB, speciesID='bb', mediaName='kombucha_media'),
        'ki': sim.createBacteria(sim.DB, speciesID='ki', mediaName='kombucha_media')
    })

    for sp in microbiome_feed.subpopD:
        microbiome_feed.subpopD[sp].count = 0

    st.sidebar.subheader("Initial Subpopulation Counts")
    for sp in microbiome.subpops:
        microbiome.subpopD[sp].count = st.sidebar.slider(f"{sp} count", 0, 1000, int(microbiome.subpopD[sp].count), 10)

    st.sidebar.subheader("Initial Media Concentrations")
    for met in metabolome.metabolites:
        metabolome.metD[met].concentration = st.sidebar.slider(f"{met}", 0.0, 50.0, float(metabolome.metD[met].concentration), 1.0)

    volume = st.sidebar.slider("Reactor Volume (L)", 5, 50, 15)

# Additional options
col1, col2 = st.columns(2)
with col1:
    show_debug = st.checkbox("Show Debug Info", value=False)
with col2:
    show_detailed_plots = st.checkbox("Show Detailed Plots", value=True)

# Run button
if "simulation_ran" not in st.session_state:
    st.session_state.simulation_ran = False

if st.button("🚀 Run Simulation", type="primary"):
    st.session_state.simulation_ran = True

# Run simulation
if st.session_state.simulation_ran:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Plots", "🔍 Analysis", "🐛 Debug"])

    with st.spinner("Running simulation..."):
        try:
            if sim_type == "Direct Reactor ODE Simulation":

                # Use your run_direct_reactor_simulation function here
                # Prepare subpopulation dict and media dict for the call
                subpop_dict = {sp: microbiome.subpopD[sp].count for sp in microbiome.subpopD}
                media_dict = {met: metabolome.metD[met].concentration for met in metabolome.metabolites}

                fig = sim.run_direct_reactor_simulation(subpop_dict, media_dict, volume=volume)

                with tab1:
                    st.success("Direct Reactor ODE Simulation Completed")
                    st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.json({
                        "Final Volume": float(fig.data[0].x[-1]) if fig.data else None,  # Example - customize as needed
                        "Note": "Add your debug info here if needed"
                    })

            elif sim_type == "Environment Test":
                env = sim.KombuchaGym(max_steps=min(user_params['param_max_steps'], 50))
                obs, _ = env.reset()
                with tab1:
                    st.success("Environment created successfully!")
                    st.write(f"**Observation Space:** {env.observation_space}")
                    st.write(f"**Action Space:** {env.action_space}")
                    st.write(f"**Initial State:** {obs}")

                with tab4:
                    if show_debug:
                        st.subheader("Debug Info")
                        st.json({
                            "current_step": env.current_step,
                            "dilution": env.dilution,
                            "volume": env.volume,
                            "metabolites": env.reactor.metabolome.concentration,
                            "subpops": {k: v.count for k, v in env.reactor.microbiome.subpopD.items()}
                        })

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

                df = pd.DataFrame(states_data)
                df.columns = [f"State_{i}" for i in range(df.shape[1])]
                df['Step'] = range(len(df))

                with tab1:
                    st.subheader("State Analysis Results")
                    st.dataframe(df.describe())

                with tab2:
                    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
                    axes = axes.flatten()
                    for i in range(min(12, df.shape[1] - 1)):
                        axes[i].plot(df['Step'], df[f"State_{i}"])
                        axes[i].set_title(f"State_{i}")
                    st.pyplot(fig)

                with tab3:
                    corr = df.corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr, cmap="coolwarm")
                    ax.set_xticks(range(len(corr)))
                    ax.set_yticks(range(len(corr)))
                    ax.set_xticklabels(corr.columns, rotation=45)
                    ax.set_yticklabels(corr.columns)
                    st.pyplot(fig)

            else:  # Random Policy
                result = sim.main(**user_params)
                df = pd.DataFrame(result)

                with tab1:
                    st.subheader("Simulation Results")
                    st.dataframe(df)

                with tab2:
                    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                    axes[0].plot(df.index, df['pH'], label='pH')
                    axes[1].plot(df.index, df['ethanol'], label='Ethanol')
                    axes[2].plot(df.index, df['reward'], label='Reward')
                    for ax in axes:
                        ax.legend()
                        ax.grid(True)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during simulation: {e}")
