from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from Data_Scoring.Attribute_Detection.Functions import detect_drop_time, detect_steady_state
import matplotlib

 
def evaluate_voltage_control (busbar):
    

    ############### sim_data Column Headers ###############
  
    csv_path = Path(r"C:\Users\james\OneDrive\MSc Project\results") / f"{busbar}.csv"
    print(csv_path)
    sim_data = pd.read_csv(csv_path, header=0, skiprows=[1])
    print(f'alligned path to sim_data variable')

    
    ############### Ideal Results Creation ###############
    
    # The ideal results are calculated based on the RMS simulation data, so the time the simulation lasts can be changed and the script will still work

    # Extract time steps and POC values
    time_steps = sim_data['All calculations']

    poc_values = sim_data[busbar]
    print('Matching excel data to internal dataframe')
    
    # Initialize the ideal results DataFrame
    ideal_results = pd.DataFrame({'All calculations (Ideal)': time_steps, busbar + '(Ideal)': np.nan})
    print(f'removing NaN')

    print(f'finding drop time')
    # Detect the drop time in the original simulation data
    drop_time_index = detect_drop_time(poc_values)
    
    if drop_time_index is not None:
        drop_time = time_steps.iloc[drop_time_index]
        print(f"Drop detected at time {drop_time} s")
        
        # Determine the steady state condition and set the final value accordingly
        last_half_avg = poc_values[int(len(poc_values) * 0.5):].mean()
        print(f"Average of the last half of the POC values: {last_half_avg}")
        
        
        # Set the POC value just before the drop to 1
        ideal_results.loc[ideal_results['All calculations (Ideal)'] <= drop_time, busbar + '(Ideal)'] = 1
        
        # here I am finding the min value so I can filter out all data before the min value and give that to the steady state function as if I dont do this it will see the time before voltage drop as steady state
        # also I need the min value to see if its within range and the ideal results need to be adjusted to match the actual fall rate
        # Step 1: Filtering
        filtered_poc_values = sim_data[(sim_data[busbar] < 1) & (sim_data[busbar] <= last_half_avg)]

        # Step 2: Finding the Minimum
        min_value = filtered_poc_values[busbar].min()
        print(f"Minimum value in the filtered POC data: {min_value}")

        # Assuming 'min_value' is the minimum POC value after which you want to consider data
        min_value_index = sim_data[sim_data[busbar] == min_value].index[0]
        print('min value index' + str(min_value_index))
        min_value_time = time_steps.iloc[min_value_index]
        print('min value time' + str(min_value_time))
        filtered_poc_values = sim_data.loc[min_value_index:]

        # To print the first 1000 rows of the DataFrame for debugging purposes
        print(filtered_poc_values[busbar].head(1000))


        # Detect the steady state in the original simulation data
        steady_state_index = detect_steady_state(filtered_poc_values[busbar], last_half_avg)
        print('Steady state index:')
        print(steady_state_index)
        
        if 0.92 < min_value < 1 :
            print('min value within range, adjusting ideal')
            # here I am adjusting the fall rate of the ideal results to match the actual fall rate, so I dont badly score a good result
                    # Calculate the absolute difference from drop_time + 0.1 for all time steps
            time_differences = abs(ideal_results['All calculations (Ideal)'] - min_value_time)

            # Find the index of the minimum difference
            closest_index = time_differences.idxmin()
            for i, value in enumerate(poc_values):
                if min_value < value < 0.98:
                    break

            first_interpolate_value = ideal_results.at[i, 'All calculations (Ideal)']

            # Set the value 0.94 at the closest time step
            ideal_results.at[min_value_index, busbar + '(Ideal)'] = min_value

            voltage_descent = ideal_results[ideal_results['All calculations (Ideal)'] >= first_interpolate_value].index.min()

            # Step 2: Set values to NaN within this range (excluding the start and end points)
            ideal_results.loc[(ideal_results.index > voltage_descent) & (ideal_results.index < min_value_index), busbar + '(Ideal)'] = np.nan

            # Step 3: Perform linear interpolation
            ideal_results[busbar + '(Ideal)'] = ideal_results[busbar + '(Ideal)'].interpolate(method='linear')
            
        else:
            print('setting drop to 0.94')
                    # Calculate the absolute difference from drop_time + 0.1 for all time steps
            time_differences = abs(ideal_results['All calculations (Ideal)'] - (drop_time + 0.1))

            # Find the index of the minimum difference
            closest_index = time_differences.idxmin()

            # Set the value 0.94 at the closest time step
            ideal_results.at[closest_index, busbar + '(Ideal)'] = 0.94

            voltage_descent = ideal_results[ideal_results['All calculations (Ideal)'] >= drop_time].index.min()

            # Step 2: Set values to NaN within this range (excluding the start and end points)
            ideal_results.loc[(ideal_results.index > voltage_descent) & (ideal_results.index < closest_index), busbar + '(Ideal)'] = np.nan

            # Step 3: Perform linear interpolation
            ideal_results[busbar + '(Ideal)'] = ideal_results[busbar + '(Ideal)'].interpolate(method='linear')


        if steady_state_index is not None:
            # Adjust the steady_state_index with the starting index of filtered_poc_values
            adjusted_index = steady_state_index + min_value_index
            print(f"Steady state reached at time {time_steps.iloc[adjusted_index]} s")
            actual_rise_time = time_steps.iloc[adjusted_index] - drop_time
            print(f"Actual rise time: {actual_rise_time} s")
            # ... rest of your code where you use steady_state_index ...
            if actual_rise_time < 0.6 :
                is_steady_state = 0.95 <= last_half_avg <= 1
                final_value = last_half_avg if is_steady_state else 1
                post_final_drop_time = drop_time + actual_rise_time
                ideal_results.loc[ideal_results['All calculations (Ideal)'] >= post_final_drop_time, busbar + '(Ideal)'] = final_value
                """
                # Calculate the new slope for the ideal results to match the actual rise rate
                slope = (0.94 - 1) / actual_rise_time  # The drop is to 0.94 as per user's script
            
                # Apply this new slope to the ideal results
                for i in range(drop_time_index + 1, steady_state_index + 1):
                    time_diff = time_steps.iloc[i] - drop_time
                    ideal_results.at[i, busbar + '(Ideal)'] = 1 + slope * time_diff

                # Re-interpolate if necessary for any subsequent points
                ideal_results[busbar + '(Ideal)'] = ideal_results[busbar + '(Ideal)'].interpolate(method='linear')
                """
                            # Step 1: Identify the range for interpolation
                # Find the index for post_final_drop_time
                post_final_drop_index = ideal_results[ideal_results['All calculations (Ideal)'] >= post_final_drop_time].index.min()

                # Step 2: Set values to NaN within this range (excluding the start and end points)
                ideal_results.loc[(ideal_results.index > closest_index) & (ideal_results.index < post_final_drop_index), busbar + '(Ideal)'] = np.nan

                # Step 3: Perform linear interpolation
                ideal_results[busbar + '(Ideal)'] = ideal_results[busbar + '(Ideal)'].interpolate(method='linear')
                print("Ideal results adjusted to match the actual rise rate")
            else:
                print("Steady state not reached within the simulation time.")
                is_steady_state = 0.95 <= last_half_avg <= 1
                final_value = last_half_avg if is_steady_state else 1
                post_final_drop_time = drop_time + 0.1 + 0.5
                ideal_results.loc[ideal_results['All calculations (Ideal)'] >= post_final_drop_time, busbar + '(Ideal)'] = final_value
     
                # Handle the case where the steady state is not found
                # Step 1: Identify the range for interpolation
                # Find the index for post_final_drop_time
                post_final_drop_index = ideal_results[ideal_results['All calculations (Ideal)'] >= post_final_drop_time].index.min()

                # Step 2: Set values to NaN within this range (excluding the start and end points)
                ideal_results.loc[(ideal_results.index > closest_index) & (ideal_results.index < post_final_drop_index), busbar + '(Ideal)'] = np.nan

                # Step 3: Perform linear interpolation
                ideal_results[busbar + '(Ideal)'] = ideal_results[busbar + '(Ideal)'].interpolate(method='linear')
                print("Ideal results were not adjusted to match the actual rise rate")

    else:
        print("No significant drop detected in the POC voltage")
        print("Likely Power Factory Error")

    # Save the final ideal results to an Excel file, left as a comment as it is not needed for the script but can be used for debugging !!!!!!!!!!!
    ideal_results_file_path = (r'C:\Users\james\OneDrive\MSc Project\results_ideal\Ideal Excel Results Voltage.csv') 
    ideal_results.to_csv(ideal_results_file_path, index=False)
    
    ############### Importing Ideal Data ###############

    Ideal_Results_Table = ideal_results             #pd.read_csv(r'C:\Users\JamesThornton\source\repos\Python Dissertation Script\Python Dissertation Script\Ideal Excel Results.csv')


    # Using Genetic algorithms to find the best kp & ki values plan as of 31/12/2023 14.11
    
    
    ############### Scoring between Ideal Data and Actual Data ###############
    if 0.92 < min_value < 1 :
        x_start = min_value  # Set your starting x-value to where the simualtion starts in the time domain
    else :
        x_start = 0.94
        
    # Filter the data
    filtered_sim_data = sim_data[sim_data[busbar] >= x_start]
    filtered_ideal_data = Ideal_Results_Table[Ideal_Results_Table[busbar + '(Ideal)'] >= x_start]

    # Assuming the time and voltage columns are still correctly aligned after filtering
    #mae_time = np.abs(filtered_sim_data['All calculations'] - filtered_ideal_data['All calculations (b:tnow in s) (Ideal)']).mean()
    mae_voltage = float(np.abs(filtered_sim_data[busbar] - filtered_ideal_data[busbar + '(Ideal)']).mean())
    
    ############### Steady State Peak Ratio ###############
    
    # Load the Excel file

    # Select the column - replace 'ColumnName' with your actual column name
    column = sim_data[busbar]

    # Find the lowest number in the column
    lowest_number = column.min()
    
    filtered_sim_data_max = sim_data[sim_data['All calculations'] > lowest_number]
        
    # Find the maximum POC voltage value
    max_POC_voltage_value = filtered_sim_data_max[busbar].max()
    
    SS_Peak_Ratio_PU = (max_POC_voltage_value - final_value) / final_value
    
        
    ############### Fitness Value ###############

    Overshoot_Compensation = SS_Peak_Ratio_PU + 1
    
    if mae_voltage <= 0.00085 and max_POC_voltage_value > final_value:
        fitness_value = mae_voltage * Overshoot_Compensation
    else  :
        fitness_value = mae_voltage


    #print("Time column Mean Absolute Error (after x_start) = " + str(mae_time))
    print("Voltage column Mean Absolute Error (after x_start) = " + str(mae_voltage))
    #print("Max POC Voltage Value = " + str(max_POC_voltage_value))
    '''
    print("kp = " + str(kp))
    print("ki = " + str(ki))
    '''
    print("Fitness Value = " + str(fitness_value))
    
    matplotlib.pyplot.close('all')                                                          # Close all open figures to save memory 
    '''
    if return_gradients:
        # Approximate the gradient with respect to kp
        delta = 0.01  # Small change to apply to kp and ki
        kp_perturbed = kp + delta
        fitness_perturbed_kp = evaluate_voltage_control((ki, kp_perturbed), pf_data, return_gradients=False)  # Evaluate fitness with perturbed kp without gradients
        grad_kp = (fitness_perturbed_kp - fitness_value) / delta  # Approximate partial derivative with respect to kp

        # Approximate the gradient with respect to ki
        ki_perturbed = ki + delta
        fitness_perturbed_ki = evaluate_voltage_control((ki_perturbed, kp), pf_data, return_gradients=False)  # Evaluate fitness with perturbed ki without gradients
        grad_ki = (fitness_perturbed_ki - fitness_value) / delta  # Approximate partial derivative with respect to ki

        return fitness_value, grad_kp, grad_ki
    '''
    return (fitness_value)
