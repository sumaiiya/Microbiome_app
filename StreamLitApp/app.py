import matplotlib.pyplot as plt
import streamlit as st
import importlib.util
import os
import pandas as pd
import matplotlib
matplotlib.use("agg")
from pony.orm import db_session

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
        if spec is None or spec.loader is None:
            st.error("Could not load module spec or loader.")
            return None

        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
        return sim
    except Exception as e:
        st.error(f"Error loading simulation module: {e}")
        return None


def fix_user_params(params):
    # Map param_ keys to expected argument names
    mapping = {
        "param_max_steps": "max_steps",
        "param_reward_scale": "reward_scale",
        "param_simul_time": "simulTime",
        "param_simul_steps": "SimulSteps",
        "param_dilution": "dilution",
        "param_volume": "volume",
        #"param_max_dilution": "max_dilution",
        #"param_unit_change": "unit_change",
        #"param_target_ethanol": "param_target_ethanol",  # keep as is or adjust as needed
    }
    fixed = {}
    for k, v in params.items():
        new_key = mapping.get(k, k)
        if new_key in ["max_steps", "dilution", "volume"]:
            continue
        fixed[new_key] = v
    return fixed


# Page setup
st.set_page_config(page_title="Kombucha Simulation",
                   page_icon="🍶", layout="wide")

st.title("🍶 Microbial Simulation Interface")
st.markdown("Interactive simulation of Kombucha microbial dynamics")

sim = load_simulation_module()
if sim is None:
    st.stop()

if not getattr(sim, "ACTUAL_DB_AVAILABLE", False):
    st.error("❌ Database classes not available!")
    st.stop()

if getattr(sim, "DB", None) is None:
    st.error("❌ Database not found!")
    st.stop()

st.success("✅ Database and classes loaded successfully!")
print(sim)

# Simulation type selection
sim_type = st.selectbox(
    "Choose simulation type",
    ["Random Policy", "Environment Test",
     "State Analysis", "Direct Reactor ODE Simulation"],
    help="Select the type of simulation to run"
)

# Parameter sections
current_params = {
    "max_steps": getattr(sim, 'param_max_steps', 5),
    "reward_scale": getattr(sim, 'param_reward_scale', 100),
    "simul_time": getattr(sim, 'param_simul_time', 1),
    "simul_steps": getattr(sim, 'param_simul_steps', 5),
    "dilution": getattr(sim, 'param_dilution', 0.5),
    "volume": getattr(sim, 'param_volume', 100),
    "max_dilution": getattr(sim, 'param_max_dilution', 60),
    "unit_change": getattr(sim, 'param_unit_change', 0.01),
    # "target_ethanol": getattr(sim, 'param_target_ethanol', 10.0)
}

# Sidebar parameters
user_params = {}

if sim_type == "Random Policy":
    st.sidebar.header("Gym Simulation Parameters")

    user_params['param_max_steps'] = st.sidebar.slider(
        "Max Steps", 5, 20, current_params['max_steps'])
    user_params['param_dilution'] = st.sidebar.slider(
        "Dilution", 0.0, 2.0, current_params['dilution'], 0.1)
    user_params['param_volume'] = st.sidebar.slider(
        "Volume (L)", 50, 500, current_params['volume'])  # fixed key here
    user_params['param_unit_change'] = st.sidebar.slider(
        "Unit Change", 0.001, 0.1, current_params['unit_change'], 0.001, format="%.3f")
    st.sidebar.subheader("🎯 Optimization Goals")

    try:
        temp_env = sim.KombuchaGym(**fix_user_params(user_params))
    except Exception as e:
        st.sidebar.error(f"Could not initialize environment for goal setup: {e}")
        temp_env = None

    optimization_goals = {"metabolites": {}, "subpopulations": {}}

    if temp_env:
        goal_type = st.sidebar.selectbox(
            "Select Goal Type",
            ["Target Metabolite Concentrations", "Target Subpopulations"]
        )

        if goal_type == "Target Metabolite Concentrations":
            default_met_targets = {
                name: temp_env.reactor.metabolome.metD[name].concentration
                for name in temp_env.reactor.metabolome.metabolites
            }
            for met, default in default_met_targets.items():
                with st.sidebar.expander(f"{met}"):
                    choice = st.radio(
                        f"Goal type for {met}",
                        options=["None", "Min", "Max"],
                        index=0,
                        key=f"{met}_choice"
                    )

                    min_val = (choice == "Min")
                    max_val = (choice == "Max")

                    target_val = st.number_input(
                        f"{met} target",
                        value=0.0,
                        step=0.1,
                        key=f"{met}_target"
                    )

                    optimization_goals["metabolites"][met] = {
                        "min": min_val,
                        "max": max_val,
                        "target": target_val
                    }


        elif goal_type == "Target Subpopulations":
            default_subpop_targets = {
                name: sub.count for name, sub in temp_env.reactor.microbiome.subpopD.items()
            }
            for subp, default in default_subpop_targets.items():
                with st.sidebar.expander(f"{subp}"):
                    choice = st.radio(
                        f"Goal type for {subp}",
                        options=["None", "Min", "Max"],
                        index=0,
                        key=f"{subp}_choice"
                    )

                    min_val = (choice == "Min")
                    max_val = (choice == "Max")

                    target_val = st.number_input(
                        f"{subp} target",
                        value=0.0,
                        step=0.1,
                        key=f"{subp}_target"
                    )

                    optimization_goals["subpopulations"][subp] = {
                        "min": min_val,
                        "max": max_val,
                        "target": target_val
                    }

        

if sim_type == "Direct Reactor ODE Simulation":
    st.sidebar.header("ODE Reactor Setup")

    metabolome = sim.createMetabolome(
        sim.DB, mediaName='kombucha_media', pHFunc=sim.getpH)
    metabolome_feed = sim.createMetabolome(
        sim.DB, mediaName='kombucha_media', pHFunc=sim.getpH)

    with db_session:
        species_ids = [row[0] for row in sim.DB.execute("SELECT id FROM species") or []]

        microbio_dict = {}
        for sp_id in species_ids:
            try:
                microbio_dict[sp_id] = sim.createBacteria(sim.DB, speciesID=sp_id, mediaName='kombucha_media')
            except Exception as e:
                st.warning(f"⚠️ Could not create bacteria for species '{sp_id}': {e}")

        microbiome = sim.Microbiome(microbio_dict)

        microbio_feed_dict = {}
        for sp_id in species_ids:
            try:
                feed_bac = sim.createBacteria(sim.DB, speciesID=sp_id, mediaName='kombucha_media')
                for subp in feed_bac.subpopulations.values():
                    subp.count = 0
                microbio_feed_dict[sp_id] = feed_bac
            except Exception as e:
                st.warning(f"⚠️ Could not create feed bacteria for '{sp_id}': {e}")

        microbiome_feed = sim.Microbiome(microbio_feed_dict)

    st.sidebar.subheader("Initial Subpopulation Counts")
    for sp in microbiome.subpops:
        microbiome.subpopD[sp].count = st.sidebar.slider(
            f"{sp} count", min_value=0.0, max_value=10.0,
            value=float(microbiome.subpopD[sp].count), step=0.5
        )

    st.sidebar.subheader("Initial Media Concentrations")
    for met in metabolome.metabolites:
        metabolome.metD[met].concentration = st.sidebar.slider(
            f"{met} concentration",
            min_value=0.0,
            max_value=10.0,
            value=float(metabolome.metD[met].concentration),
            step=0.5,
        )

    volume = st.sidebar.slider(
        "Reactor Volume (L)", min_value=5, max_value=50, value=15, step=1)

col1, col2 = st.columns(2)
with col1:
    show_debug = st.checkbox("Show Debug Info", value=False)
with col2:
    show_detailed_plots = st.checkbox("Show Detailed Plots", value=True)

if "simulation_ran" not in st.session_state:
    st.session_state.simulation_ran = False

if st.button("🚀 Train Policy", type="primary"):
    st.session_state.simulation_ran = True

if st.session_state.simulation_ran:
    optimization_goals = {}
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Results", "📈 Plots", "🔍 Analysis", "🐛 Debug"])

    with st.spinner("Running simulation..."):
        try:
            if sim_type == "Direct Reactor ODE Simulation":
                subpop_dict = {
                    sp: microbiome.subpopD[sp].count for sp in microbiome.subpopD}
                media_dict = {
                    met: metabolome.metD[met].concentration for met in metabolome.metabolites}

                fig = sim.run_direct_reactor_simulation(
                    subpop_dict, media_dict, volume=volume)

                with tab1:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Reactor Simulation Completed")

                with tab4:
                    st.json({
                        "Final Volume": float(fig.data[0].x[-1]) if fig.data else None,
                        "Note": "Add your debug info here if needed"
                    })

            elif sim_type in ["Environment Test", "State Analysis", "Random Policy"]:
                fixed_params = fix_user_params(user_params)
                env = sim.KombuchaGym(**fixed_params)
                #print("Environment", env)
                #print(vars(env))  # prints a dict of the env instance's attributes (if any)

                # Or, to explore specific parts, e.g. reactor state:
                #print("Reactor:", env.reactor)
                obs, _ = env.reset()
                pH = env.reactor.pH
                fixed_params["optimization_goals"] = optimization_goals

                if sim_type == "Environment Test":
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
                    states_data = []
                    states_data.append(obs.copy())

                    for i in range(3):
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
                    result = sim.main(**fixed_params)

                    if isinstance(result, list) and isinstance(result[0], dict):
                        df = pd.DataFrame(result)
                        st.success("Simulation completed.")
                    else:
                        st.error("Unexpected result format. Could not convert to DataFrame.")
                        st.write("Raw result:", result)

                    with tab1:
                        st.subheader("Simulation Results")
                        st.dataframe(df)

                    with tab2:
                        if 'reward' in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(df.index, df['reward'], label='Reward')
                            ax.set_title('Reward Over Time')
                            ax.set_xlabel('Step')
                            ax.set_ylabel('Reward')
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.warning("Reward data not available.")


        except Exception as e:
            st.error(f"Error during simulation: {e}")
