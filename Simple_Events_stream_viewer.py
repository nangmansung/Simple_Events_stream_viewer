import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime
import os

# --- Setup for Demonstration ---
# This part creates a dummy data file if it doesn't exist.
# This is necessary for the code to be runnable.
# In your real use case, you can replace DUMMY_FILE_PATH with your actual file path.
DUMMY_FILE_PATH = "./Drive_with_Dark_BA.npy"


if not os.path.exists(DUMMY_FILE_PATH):
    print("Dummy data file not found. Creating a new one for demonstration.")
    # Create a dummy dataset with shape (t, x, y, p, l)
    dummy_events = 50000
    dummy_data = np.zeros((dummy_events, 5))
    dummy_data[:, 0] = np.sort(np.random.randint(0, 1_000_000, dummy_events)) # Timestamps
    dummy_data[:, 1] = np.random.randint(0, 200, dummy_events) # x coordinates
    dummy_data[:, 2] = np.random.randint(0, 150, dummy_events) # y coordinates
    dummy_data[:, 3] = np.random.randint(0, 2, dummy_events)   # Polarity
    dummy_data[:, 4] = np.random.randint(0, 2, dummy_events)   # Label
    np.save(DUMMY_FILE_PATH, dummy_data)
    print(f"'{DUMMY_FILE_PATH}' created.")

# --- Load Data ---
try:
    original_data = np.load(DUMMY_FILE_PATH) # (t,x,y,p,l)
except FileNotFoundError:
    st.error(f"Error: The data file was not found at '{DUMMY_FILE_PATH}'.")
    st.stop()


class Display_Control:
    # English Comment: Added 'data' to the constructor to pass the loaded numpy array.
    def __init__(self, width, height, data):
        self.height = height
        self.width = width
        self.original_label = np.zeros((height, width))
        self.original_TS = np.zeros((height, width))
        self.original_P = np.zeros((height, width))
        
        # English Comment: This grid stores the state before the latest events were added.
        self.previous_label_grid = np.zeros((height, width))

        self.TW = 100_000 # 100ms
        self.event_idx = 0
        self.now_TS = 0
        # English Comment: Store the raw event data.
        self.data = data

    def change_TS(self, new_TW):
        # English Comment: Changes the time window for event accumulation.
        self.TW = new_TW
    
    def clear(self):
        # English Comment: Resets the current event grids.
        self.original_label.fill(0)
        self.original_TS.fill(0)
        self.original_P.fill(0)
    
    def data_crop(self):
        # English Comment: Crops the data based on the current time and time window.
        current_idx = self.event_idx
        if current_idx >= len(self.data):
            current_idx = len(self.data) - 1
            add_log("Reached the end of the data.")
            
        current_time = self.data[current_idx][0]
        start_idx = np.searchsorted(self.data[:, 0], current_time - self.TW, side='left')
        return self.data[start_idx:current_idx+1]
    
    def next_events(self, event_num):
        # English Comment: Save the current grid as the previous state before updating.
        self.previous_label_grid = self.original_label.copy()
        
        self.clear()
        self.event_idx = self.event_idx + int(event_num)

        cropped_original_data = self.data_crop()

        for t,x,y,p,l in cropped_original_data:
            if 0 <= y < self.height and 0 <= x < self.width:
                y, x = int(y), int(x)
                if l==1:
                    self.original_label[y,x] = 1 # Real event
                elif l==0:
                    self.original_label[y,x] = -1 # Noise event
                self.original_TS[y,x] = t
        
        # English Comment: Return the newly updated grid.
        return self.original_label

# --- Set page configuration to wide mode ---
st.set_page_config(layout="wide")

# --- Initialize session state ---
if 'grid_generated' not in st.session_state:
    st.session_state.grid_generated = False
    st.session_state.height = 260
    st.session_state.width = 346
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'main_display' not in st.session_state:
    st.session_state.main_display = None
if 'display_matrix' not in st.session_state:
    st.session_state.display_matrix = None
# English Comment: State variable to toggle the highlight view.
if 'highlight_mode' not in st.session_state:
    st.session_state.highlight_mode = False

# --- Helper function to add logs ---
def add_log(message):
    """Adds a timestamped message to the log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.insert(0, f"[{timestamp}] {message}")

# --- App Title ---
st.title("Event stream visualize")
st.write(f"File : {DUMMY_FILE_PATH}")

# --- Create two columns with a 8:2 width ratio ---
left_column, right_column = st.columns([8, 2])

# --- Right Column: Input Controls ---
with right_column:
    st.subheader("⚙️ 1. Display create")
    
    height_input = st.number_input("Height", min_value=1, max_value=1000, value=st.session_state.height, step=1)
    width_input = st.number_input("Width", min_value=1, max_value=1000, value=st.session_state.width, step=1)
    
    if st.button("Done", key="done_button", use_container_width=True):
        st.session_state.grid_generated = True
        st.session_state.height = height_input
        st.session_state.width = width_input
        add_log(f"Display set {width_input} * {height_input}")
        w = int(width_input)
        h = int(height_input)
        
        st.session_state.main_display = Display_Control(w, h, original_data)
        st.session_state.display_matrix = np.zeros((h, w))
        st.session_state.highlight_mode = False # Reset highlight mode on init
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Event Controls")

    events_col_input, events_col_button = st.columns([3, 1])
    with events_col_input:
        next_events_val = st.number_input("next events", min_value=1, value=1000, step=100, label_visibility="collapsed")
    with events_col_button:
        if st.button("Go", key="events_ok", use_container_width=True):
            if st.session_state.main_display:
                add_log(f"Processing next {next_events_val} events.")
                new_matrix = st.session_state.main_display.next_events(next_events_val)
                st.session_state.display_matrix = new_matrix
            else:
                add_log("Error: Please click 'Done' to initialize the display first.")

    # English Comment: This button toggles the view for new events.
    if st.button("Highlight New / Real-Noise View", key="highlight_button", use_container_width=True):
        if st.session_state.main_display:
            st.session_state.highlight_mode = not st.session_state.highlight_mode
            mode = "Highlight" if st.session_state.highlight_mode else "Real/Noise"
            add_log(f"Switched to {mode} view.")
        else:
            add_log("Error: Please click 'Done' to initialize the display first.")

    store_col_input, store_col_button = st.columns([3, 1])
    with store_col_input:
        events_store_val = st.number_input("events_store (µs)", min_value=1, value=100000, label_visibility="collapsed")
    with store_col_button:
        if st.button("ok", key="store_ok", use_container_width=True):
            if st.session_state.main_display:
                add_log(f"events_store changed : {events_store_val}")
                st.session_state.main_display.change_TS(int(events_store_val))
            else:
                add_log("Error: Please click 'Done' to initialize the display first.")

    st.markdown("---")
    st.subheader("📝 Action Log")
    log_display_text = "\n".join(st.session_state.log_messages)
    st.text_area("Log entries", value=log_display_text, height=200, disabled=True, label_visibility="collapsed")


# --- Left Column: Grid Visualization ---
with left_column:
    st.subheader("🖼️ Grid Display")
    
    if st.session_state.display_matrix is not None:
        try:
            # English Comment: Check if highlight mode is active.
            if st.session_state.highlight_mode:
                # --- Highlight Mode Plotting (Old vs New) ---
                current_grid = st.session_state.display_matrix
                previous_grid = st.session_state.main_display.previous_label_grid
                
                # Create a new grid for display: 0=background, 1=old, 2=new
                highlight_grid = np.zeros(current_grid.shape, dtype=int)
                
                # Mark old events as 1 (black)
                highlight_grid[previous_grid != 0] = 1
                # Mark new events as 2 (orange)
                highlight_grid[(current_grid != 0) & (previous_grid == 0)] = 2

                # Define custom colors: 0=white, 1=black, 2=orange
                cmap_highlight = ListedColormap(['#FFFFFF', '#000000', '#FFA500'])
                
                matrix_to_plot = highlight_grid
                cmap_to_use = cmap_highlight
                vmin, vmax = 0, 2
            
            else:
                # --- Real/Noise Mode Plotting ---
                # MODIFICATION: Use Black for Real (1) and Orange for Noise (-1).
                source_matrix = st.session_state.display_matrix
                
                # Create a matrix for plotting with indices for the colormap.
                # 0: Background (White)
                # 1: Noise (Orange)
                # 2: Real (Black)
                matrix_to_plot = np.full(source_matrix.shape, 0, dtype=int) # Default to background
                matrix_to_plot[source_matrix == -1] = 1 # Set index for Noise events
                matrix_to_plot[source_matrix == 1] = 2  # Set index for Real events

                # English Comment: Define colormap: White (background), Orange (noise), Black (real).
                cmap_to_use = ListedColormap(['#FFFFFF', '#FFA500', '#000000'])
                vmin, vmax = 0, 2


            # --- Common Plotting Code ---
            grid_height, grid_width = matrix_to_plot.shape
            aspect_ratio = grid_height / grid_width if grid_width > 0 else 1
            fig_width = 12.0
            fig_height = fig_width * aspect_ratio

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.imshow(matrix_to_plot, cmap=cmap_to_use, interpolation='none', vmin=vmin, vmax=vmax)
            
            ax.set_xticks(np.arange(-.5, grid_width, 1), minor=False)
            ax.set_yticks(np.arange(-.5, grid_height, 1), minor=False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
            
            for spine in ax.spines.values():
                spine.set_visible(False)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while creating the grid: {e}")
    else:
        st.info("Grid will be displayed here after you click 'Done'.")