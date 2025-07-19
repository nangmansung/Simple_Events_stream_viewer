# Event Stream Visualizer

A simple Streamlit viewer for real-time event stream data.

Thank you for Gemini 2.5 pro
<img width="1901" height="961" alt="Image" src="https://github.com/user-attachments/assets/9ee48cbc-8c40-4109-897e-6cc4f76f6c2b" />

## 🌟 About This Project

This application provides an interactive way to visualize event-based sensor data stored in a NumPy (`.npy`) file. It allows users to step through event data, distinguish between 'real' and 'noise' events, and highlight newly occurring events within a specified time window.

## ✨ Features

-   **Load Data**: Loads event data from a local `.npy` file. <- You should do it at ".py file"
-   **Customizable Display**: Set the height and width of the visualization grid.
-   **Event Navigation**: Process and view a specific number of new events at a time. <- [GO] button
-   **Time Window Control**: Adjust the time duration (in microseconds) for accumulating events to display. <- [OK] button
-   **Dual View Modes**:
    1.  **Real/Noise View**: Displays 'real' events in black and 'noise' events in orange.
    2.  **Highlight View**: Shows previously existing events in black and newly added events in orange.
-   **Action Log**: Keeps a timestamped log of all user actions for easy tracking.

## 🚀 How to Run

1.  **Prerequisites**:
    * Python 3.x
    * A NumPy data file (`.npy`) with event data. If you don't have one, the script will automatically create a dummy file named `Drive_with_Dark_BA.npy` for demonstration purposes.

2.  **Install necessary libraries**:
    ```bash
    pip install streamlit numpy matplotlib
    ```

3.  **Run the application**:
    Save the code as a Python file (e.g., `app.py`) and run the following command in your terminal:
    ```bash
    streamlit run Simple_Events_stream_viewer.py
    ```

4.  **Open your browser**:
    Streamlit will open a new tab in your web browser with the running application.

## 📁 Data Format

The application expects the `.npy` file to contain an array where each row represents an event with the following structure:

-   `t`: Timestamp (integer, in microseconds)
-   `x`: X-coordinate (integer)
-   `y`: Y-coordinate (integer)
-   `p`: Polarity (0 or 1) <- it does not used in ".py". Maybe later added. Now just for place hold
-   `l`: Label (1 for a real event, 0 for a noise event)
