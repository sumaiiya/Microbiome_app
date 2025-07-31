import sys
import matplotlib.pyplot as plt
import streamlit as st
import importlib.util
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
import traceback
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import warnings
import kombucha_simulation as sim

# Suppress specific warnings that might clutter the output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Kombucha Simulation",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_simulation_module():
    """Load the kombucha simulation module with better error handling"""
    try:
        module_path = os.path.join(BASE_DIR, "kombucha_simulation.py")

        st.write(f"🔍 Looking for module at: {module_path}")
        st.write(f"🔍 File exists: {os.path.exists(module_path)}")

        if not os.path.exists(module_path):
            st.error(f"❌ Module not found at: {module_path}")
            st.info("Please ensure 'kombucha_simulation.py' is in the same directory as this app.")
            return None

        spec = importlib.util.spec_from_file_location("kombucha_simulation", module_path)
        if spec is None or spec.loader is None:
            st.error("❌ Could not load module spec or loader.")
            return None

        st.write("🔍 Loading module...")

        # Capture stdout during module loading to show database connection status
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        output = f.getvalue()
        if output:
            st.text("Module loading output:")
            st.code(output)

        st.write("✅ Module loaded")

        # Check if required components are available
        db_available = getattr(module, 'ACTUAL_DB_AVAILABLE', False)
        db_object = getattr(module, 'DB', None)

        st.write(f"🔍 Database available: {db_available}")
        st.write(f"🔍 Database object: {'✅ Connected' if db_object else '❌ Not found'}")

        if not db_available:
            st.error("❌ Database not available in simulation module")
            st.write("This might be due to:")
            st.write("- Missing database file")
            st.write("- Incorrect database path")
            st.write("- Database connection issues")
            return None

        if db_object is None:
            st.error("❌ Database connection not found")
            return None

        return module

    except Exception as e:
        st.error(f"❌ Failed to load kombucha_simulation.py: {e}")
        with st.expander("Show detailed error"):
            st.text(traceback.format_exc())
        return None

def test_simulation_safety(sim):
    """Test the simulation with safety checks"""
    try:
        st.write("🔍 Testing simulation safety...")

        # Test with minimal parameters
        test_params = {
            'param_max_steps': 1,
            'simulTime': 0.1,
            'SimulSteps': 2,
            'param_dilution': 0.1,
            'param_volume': 50,
            'run_simple_test': True
        }

        # Run test with timeout
        def run_test():
            return sim.main(**test_params)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_test)
            try:
                result = future.result(timeout=15)  # 15 second timeout

                if isinstance(result, list) and len(result) > 0:
                    st.success("✅ Basic simulation test passed!")
                    return True, result
                else:
                    st.warning("⚠️ Simulation returned unexpected result")
                    return False, result

            except TimeoutError:
                st.error("❌ Simulation test timed out")
                return False, None
            except Exception as e:
                st.error(f"❌ Simulation test failed: {e}")
                return False, None

    except Exception as e:
        st.error(f"❌ Error in safety test: {e}")
        return False, None

def create_safe_parameter_ui():
    """Create parameter UI with safe defaults"""
    st.sidebar.header("⚙️ Simulation Parameters")

    # Use more conservative defaults
    params = {}

    params['param_max_steps'] = st.sidebar.slider(
        "Max Steps",
        min_value=1,
        max_value=5,
        value=2,  # Reduced default
        help="Number of simulation steps (lower = faster)"
    )

    params['simulTime'] = st.sidebar.slider(
        "Simulation Time",
        min_value=0.1,
        max_value=2.0,
        value=0.5,  # Reduced default
        step=0.1,
        help="Duration of each simulation step"
    )

    params['SimulSteps'] = st.sidebar.slider(
        "Integration Steps",
        min_value=2,
        max_value=10,
        value=3,  # Reduced default
        help="Number of integration steps per simulation step"
    )

    params['param_dilution'] = st.sidebar.slider(
        "Dilution Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.3,  # Conservative default
        step=0.1,
        help="Rate of medium dilution"
    )

    params['param_volume'] = st.sidebar.slider(
        "Volume (L)",
        min_value=10,
        max_value=100,
        value=50,  # Reduced default
        help="Reactor volume"
    )

    # Advanced parameters in expander
    with st.sidebar.expander("🔧 Advanced Settings", expanded=False):
        params['param_unit_change'] = st.sidebar.slider(
            "Unit Change",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            help="Step size for parameter changes"
        )

        params['param_max_dilution'] = st.sidebar.slider(
            "Max Dilution",
            min_value=10,
            max_value=60,
            value=30,  # Reduced default
            help="Maximum allowed dilution rate"
        )

    # Always run simple test for safety
    params['run_simple_test'] = True

    return params

def create_multi_optimization_goals_ui(available_states_dict):
    optimization_goals = {}
    st.sidebar.subheader("🎯 Optimization Goals")

    if not isinstance(available_states_dict, dict):
        st.warning("⚠️ Available states not in dictionary format. Skipping.")
        return {}

    for state_name in available_states_dict.keys():
        include = st.sidebar.checkbox(f"Optimize {state_name}", key=f"include_{state_name}")
        if include:
            min_val = st.sidebar.number_input(f"{state_name} Min", key=f"{state_name}_min")
            max_val = st.sidebar.number_input(f"{state_name} Max", key=f"{state_name}_max")
            target_val = st.sidebar.number_input(f"{state_name} Target", key=f"{state_name}_target")
            optimization_goals[state_name] = {
                "min": min_val,
                "max": max_val,
                "target": target_val
            }

    return optimization_goals


   



def display_results_safely(result):
    """Safely display simulation results"""
    try:
        if not result or len(result) == 0:
            st.warning("⚠️ No results to display")
            return

        # Convert to DataFrame
        df = pd.DataFrame(result)

        # Check for error column
        if 'error' in df.columns:
            st.error("❌ Simulation encountered errors:")
            for i, row in df.iterrows():
                if pd.notna(row.get('error')):
                    st.error(f"Step {row.get('step', i)}: {row['error']}")

        # Display basic info
        st.subheader("📊 Simulation Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Steps Completed", len(df))

        with col2:
            if 'pH' in df.columns:
                final_pH = df['pH'].iloc[-1] if len(df) > 0 else 0
                st.metric("Final pH", f"{final_pH:.2f}")

        with col3:
            if 'volume' in df.columns:
                final_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
                st.metric("Final Volume", f"{final_volume:.1f}L")

        # Show data table
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)

        # Create plots if we have numeric data
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) > 0:
            st.subheader("📈 Visualization")

            # Filter out step column for plotting
            plot_columns = [col for col in numeric_columns if col not in ['step']]

            if len(plot_columns) > 0:
                # Create tabs for different plot types
                tab1, tab2 = st.tabs(["📊 Time Series", "📋 Data Summary"])

                with tab1:
                    if 'step' in df.columns:
                        # Plot up to 5 most interesting columns
                        cols_to_plot = plot_columns[:5]

                        for col in cols_to_plot:
                            if col in df.columns:
                                st.line_chart(df.set_index('step')[col], use_container_width=True)
                    else:
                        st.info("No step column found for time series plotting")

                with tab2:
                    # Show statistical summary
                    st.write("Statistical Summary:")
                    st.dataframe(df[numeric_columns].describe(), use_container_width=True)

        # Download option
        st.subheader("💾 Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="kombucha_simulation_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error displaying results: {e}")
        st.write("Raw result data:")
        st.json(result if isinstance(result, (dict, list)) else str(result))

def main():
    """Main application function with enhanced error handling"""

    # Title and description
    st.title("🍶 Microbial Environment Control")
    st.markdown("""
    Interactive simulation of microbial dynamics.

    
    """)

    # Debug information
    with st.expander("🔧 System Information", expanded=False):
        #st.write(f"Python version: {sys.version}")
        #st.write(f"Working directory: {os.getcwd()}")
        #st.write(f"Script directory: {BASE_DIR}")
        #st.write(f"Available files: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'Directory not found'}")

    # Load simulation module
    #st.header("🔄 Module Loading")
    #with st.spinner("Loading simulation module..."):
        sim = load_simulation_module()

    if sim is None:
        st.error("❌ Cannot proceed without simulation module")
        st.stop()

    st.success("✅ Simulation module loaded successfully!")

    # Test simulation safety
    st.header("🧪 Safety Check")
    with st.spinner("Running safety tests..."):
        safety_passed, test_result = test_simulation_safety(sim)

    if not safety_passed:
        st.error("❌ Safety tests failed. The simulation may not work correctly.")
        if test_result:
            st.write("Test result:")
            st.json(test_result)
        st.stop()

    st.success("✅ Safety tests passed!")

    # Main interface
    st.header("🎮 Simulation Interface")

    # Parameter configuration
    params = create_safe_parameter_ui()

    # Example list of states for optimization — replace with dynamic if you can
    #available_states = ['pH', 'ethanol', 'acetate', 'glucose', 'volume']
    try:
        st.write("Before calling get_dynamic_states()")
        available_states = sim.get_dynamic_states(
            param_max_steps=2,
            simulTime=0.5,
            SimulSteps=3,
            param_dilution=0.1,
            param_volume=50,
            run_simple_test=False
        )
        st.write("After calling get_dynamic_states()")
        st.write("Raw available_states type:", type(available_states))
        st.write("Raw available_states value:", available_states)

        # ✅ Normalize output to dict format
        if isinstance(available_states, list):
            available_states = {s: {} for s in available_states}
        elif isinstance(available_states, np.ndarray):
            available_states = {str(s): {} for s in available_states}
        elif not isinstance(available_states, dict):
            available_states = {}

        st.success(f"✅ Loaded dynamic states: {list(available_states.keys())}")
    except Exception as e:
        st.error(f"❌ Failed to load dynamic states: {e}")
        available_states = {s: {} for s in ['pH', 'ethanol', 'acetate', 'glucose', 'volume']}
        st.info("⚠️ Using fallback default states.")




    # Optimization goals sidebar UI
    optimization_goals = create_multi_optimization_goals_ui(available_states)

    # Show chosen optimization goals in main page for debugging
    with st.expander("🎯 Current Optimization Goals", expanded=False):
        st.json(optimization_goals)

    # Optionally merge optimization goals into simulation parameters if your sim accepts it
    params['optimization_goals'] = optimization_goals

    # Show current configuration
    with st.expander("📋 Current Configuration", expanded=False):
        st.json(params)

    # Run simulation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Run Simulation",
            type="primary",
            use_container_width=True,
            help="Click to start the simulation with current parameters"
        )

    if run_button:
        st.header("⚡ Running Simulation")

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing simulation...")
            progress_bar.progress(0.1)

            # Run simulation with timeout
            def run_simulation():
                return sim.main(**params)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_simulation)

                # Update progress while waiting
                for i in range(10):
                    time.sleep(0.5)
                    progress_bar.progress(0.1 + (i * 0.08))
                    status_text.text(f"Running simulation... {i+1}/10")

                    if future.done():
                        break

                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    progress_bar.progress(1.0)
                    status_text.text("✅ Simulation completed!")

                    # Display results
                    st.header("📊 Results")
                    display_results_safely(result)

                except TimeoutError:
                    st.error("❌ Simulation timed out after 30 seconds")
                    st.info("Try reducing the number of steps or simulation time")

                except Exception as e:
                    st.error(f"❌ Simulation failed: {e}")
                    with st.expander("Show detailed error"):
                        st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            with st.expander("Show detailed error"):
                st.text(traceback.format_exc())

        finally:
            progress_bar.empty()
            status_text.empty()

    # Additional information
    #st.header("ℹ️ Help & Information")

    #with st.expander("📖 How to Use", expanded=False):
        #st.markdown("""
        #1. **Configure Parameters**: Use the sidebar to adjust simulation parameters
        #2. **Safety First**: The app automatically runs with safe, conservative settings
        #3. **Run Simulation**: Click the "Run Simulation" button to start
        #4. **View Results**: Explore the results table and visualizations
        #5. **Download Data**: Use the download button to save results as CSV

        #**Tips:**
        #- Start with small parameter values for faster testing
        #- Check the System Information if you encounter issues
        #- Use the error details to troubleshoot problems
        #""")

  

if __name__ == "__main__":
    main()
