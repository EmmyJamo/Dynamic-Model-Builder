from scipy.signal import find_peaks, savgol_filter
import numpy as np
from tkinter import messagebox

def detect_drop_time(poc_values, threshold=0.01, window=2):

    for i in range(len(poc_values) - window):
        current_value = poc_values.iloc[i]
        next_values = poc_values.iloc[i+1:i+1+window]
        if next_values.min() < current_value * (1 - threshold):
            print(f"Drop detected at index {i}, value: {current_value}")
            return i  # Return the index of the drop

    return None  # No significant drop detected 

def detect_steady_state(poc_values, steady_state, tolerance=0.00125):


    # Define the upper and lower bounds within which the value is considered steady state
    upper_bound = steady_state * (1 + tolerance)
    lower_bound = steady_state * (1 - tolerance)
    print('Upper bound: ')
    print(upper_bound)
    print('Lower bound: ')
    print(lower_bound)
    # Iterate over the POC values
    for i, value in enumerate(poc_values):
        if lower_bound <= value <= upper_bound:    
            print('Value: ')    
            print(value)
            return i  # Return the index as soon as a value within the tolerance is found

    return None  # If the loop completes without returning, no value met the condition


def custom_log_interpolation(drop_time, drop_value, steady_state_time, steady_state_value):
    """
    Custom interpolation using a logarithmic function with two data points.

    Parameters:
    - drop_time: Time at which the drop happens.
    - drop_value: Value at the drop time.
    - steady_state_time: Time at which steady state is reached.
    - steady_state_value: Value at steady state.

    Returns:
    - a, b, c: Parameters of the logarithmic function.
    """
    
    # We assume a form of the logarithmic function: y(t) = a + b * log(c * (t - drop_time) + 1)
    # With two points (drop_time, drop_value) and (steady_state_time, steady_state_value), we can solve for a and b
    # We choose c such that the argument of log becomes 1 at t = steady_state_time
    c = 1 / (steady_state_time - drop_time)

    # a can be directly derived from the value at drop_time
    a = drop_value
    
    # b can be derived from the value at steady_state_time using the value of a and the chosen c
    b = (steady_state_value - a) / np.log(c * (steady_state_time - drop_time) + 1)
    
    return a, b, c

# Function to calculate interpolated values using the determined parameters a, b, and c
def calculate_log_values(time_array, a, b, c, drop_time):
    """
    Calculate the logarithmic function values for a given time array based on parameters a, b, c.

    Parameters:
    - time_array: Array of time points where values need to be calculated.
    - a, b, c: Parameters of the logarithmic function.
    - drop_time: Time at which the drop happens (used to adjust the time array).

    Returns:
    - Array of values calculated using the logarithmic function.
    """
    return a + b * np.log(c * (time_array - drop_time) + 1)


def detect_oscillations(data, column_name):
    # Apply smoothing
    smoothed_data = savgol_filter(data[column_name], window_length=51, polyorder=3)
    
    # Find peaks and troughs
    peaks, _ = find_peaks(smoothed_data)
    troughs, _ = find_peaks(-smoothed_data)
    
    # Calculate amplitude and count oscillations
    amplitudes = []
    for peak, trough in zip(peaks, troughs):
        amplitudes.append(smoothed_data[peak] - smoothed_data[trough])
    
    oscillation_count = len(peaks)
    
    # Calculate the period and frequency of oscillations
    periods = np.diff(peaks)  # Assuming a constant sampling rate
    average_period = np.mean(periods)
    frequency = 1 / average_period if average_period > 0 else 0
    
    # Duration of oscillation
    oscillation_duration = data.index[peaks[-1]] - data.index[peaks[0]]
    
    return {
        "amplitudes": amplitudes,
        "count": oscillation_count,
        "frequency": frequency,
        "duration": oscillation_duration
    }

def evaluate_results(best_mae, best_kp, best_ki, control_choice):
    if control_choice == "frequency" or control_choice == "generic frequency":
        variable_best_mae = 1.08
    elif control_choice == "voltage" or control_choice == "generic voltage":    
         variable_best_mae = 0.00085

    if best_mae < variable_best_mae:
        messagebox.showinfo("Success", "MAE score within range, check graph.")
    else:
        messagebox.showwarning("Warning", f"MAE score not within range. The best MAE score was {best_mae:.4f}, "
                                           f"the best kp and ki values were {best_kp} and {best_ki} respectively.")
