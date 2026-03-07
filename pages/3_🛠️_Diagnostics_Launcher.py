import streamlit as st
import subprocess
import threading
import queue
import glob
import os
import datetime

st.set_page_config(page_title="Diagnostics Launcher", page_icon="🛠️", layout="wide")

st.title("🛠️ Diagnostics Launcher")
st.markdown("Run background diagnostic scripts and generate comprehensive LLM-ready markdown reports directly from the UI.")

# Define the available scripts
SCRIPTS = {
    "Diagnose Pipeline": {
        "file": "diagnose_pipeline.py",
        "description": "Verify the underlying structural health of the ML models.",
        "icon": "🩺"
    },
    "Run Experiments": {
        "file": "run_experiments.py",
        "description": "Test hypothesis-driven strategy modifications and generate comparison reports.",
        "icon": "🧪"
    },
    "Benchmark Assets": {
        "file": "benchmark_assets.py",
        "description": "Verify the strategy works across other asset classes (optional, for robustness).",
        "icon": "🌍"
    }
}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select Script to Run")
    selected_script = st.radio("Available Scripts:", list(SCRIPTS.keys()), format_func=lambda x: f"{SCRIPTS[x]['icon']} {x}")
    
    script_info = SCRIPTS[selected_script]
    st.info(script_info["description"])
    
    if st.button(f"Run {script_info['file']}", type="primary"):
        st.session_state.running_script = script_info['file']
        st.session_state.script_output = ""
        st.rerun()

with col2:
    st.subheader("Console Output")
    
    output_container = st.empty()
    
    if 'running_script' in st.session_state and st.session_state.running_script is not None:
        script_to_run = st.session_state.running_script
        
        with st.spinner(f"Running {script_to_run}..."):
            try:
                # Run the script and capture output
                result = subprocess.run(
                    ["python", script_to_run], 
                    capture_output=True, 
                    text=True, 
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                output = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
                output_container.code(output, language="text")
                
                if result.returncode == 0:
                    st.success(f"{script_to_run} completed successfully!")
                else:
                    st.error(f"{script_to_run} failed with return code {result.returncode}.")
                    
            except Exception as e:
                output_container.error(f"Failed to execute script: {e}")
        
        # Clear the running state so it doesn't re-run on next interaction
        st.session_state.running_script = None

st.divider()

# =============================================================================
# Markdown Report Viewer
# =============================================================================
st.subheader("📄 Recently Generated Reports (LLM Ready)")

# Find all markdown files in the benchmarks folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmarks_dir = os.path.join(base_dir, "benchmarks")

if os.path.exists(benchmarks_dir):
    md_files = glob.glob(os.path.join(benchmarks_dir, "*.md"))
    # Sort by modification time, newest first
    md_files.sort(key=os.path.getmtime, reverse=True)
    
    if md_files:
        selected_file = st.selectbox(
            "Select a report to view/copy:", 
            md_files, 
            format_func=lambda x: f"{os.path.basename(x)} (Generated: {datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')})"
        )
        
        with open(selected_file, 'r') as f:
            md_content = f.read()
            
        st.markdown(md_content)
        
        # Provide a raw code block for easy copying
        with st.expander("Raw Markdown (Click 'Copy' button in top right of box)"):
            st.code(md_content, language="markdown")
    else:
        st.info("No markdown reports found in the 'benchmarks' folder.")
else:
    st.info("The 'benchmarks' folder does not exist yet. Run a diagnostic script above to generate reports.")
