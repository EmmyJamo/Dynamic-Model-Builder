from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Dynamic_Model_Builder.Data_Scoring.Attribute_Detection.Functions import detect_drop_time, detect_steady_state
import matplotlib



def evaluate_voltage_control (individual, pf_data, return_gradients=False):
    
    ############## Setting up Power Factory Parameters ##############

    poc_busbar = pf_data.app.GetCalcRelevantObjects(pf_data.poc_busbar_name)[0]
    control_dsl = pf_data.app.GetCalcRelevantObjects(pf_data.dsl_name)[0]
    ki, kp = individual                                                                          # list of ki values to iterate, the values are created in the genetic algorithm below 
         
    ############## Setting ki and kp values in the simulation  ##############

    params = control_dsl.params                                                             # Set up parameters for Power Factory
    params[5] = kp.item() if isinstance(kp, np.ndarray) and kp.size == 1 else float(kp)
    params[6] = ki.item() if isinstance(ki, np.ndarray) and ki.size == 1 else float(ki)
    #params[5] = float(kp)                                                                        # set kp (proportional) value in Power Factory
    #params[6] = float(ki)                                                                           # set ki (integral) value in Power Factory        
    control_dsl.params = params                                                             # set parameters in Power Factory
    
    ############### Run RMS Simulation ###############

    com_sim = pf_data.app.GetFromStudyCase('ComSim')
    com_sim.Execute()                                                               

    ############### Exporting RMS Simulation Data ###############

    elmres = pf_data.app.GetFromStudyCase('All calculations.ElmRes')
    comres = pf_data.app.GetFromStudyCase('ComRes')
    comres.iopt_csel = 0
    comres.iopt_tsel = 0
    comres.iopt_locn = 1
    comres.ciopt_head = 1
    comres.pResult = elmres
    comres.f_name = r'temp\rms_sim.csv'
    comres.iopt_exp = 6
    comres.Execute()

    ############### sim_data Column Headers ###############
  
    sim_data = pd.read_csv(r'temp\rms_sim.csv')
   # sim_data = sim_data.dropna(axis=1, how='all')  # Drop columns with all NaN values
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        
    for column in sim_data:
        new_header = column.split('.')[0] + ' (' + sim_data.loc[0, column] + ')'
        sim_data = sim_data.rename(columns={column: new_header})
    sim_data = sim_data.drop(0)
    sim_data = sim_data.astype(float)

    sim_data.to_csv(r'Ex7_Simulation.csv')      

    t = sim_data['All calculations (b:tnow in s)'].to_numpy()
    v_poc = sim_data[poc_busbar.loc_name + ' (m:u1 in p.u.)'].to_numpy()
    ax.plot(t, v_poc, label=f'ki={ki}')

    ############### Plotting Processed RMS Simulation Data Into .png ###############

    ax.set_ylabel('POC Voltage (pu)')
    ax.set_xlabel('Time (s)')
    ax.legend()             
    fig.tight_layout()
    fig.savefig('Ex7_RMS_Simulation')

    
    ############### Ideal Results Creation ###############
    
    # The ideal results are calculated based on the RMS simulation data, so the time the simulation lasts can be changed and the script will still work

    # Extract time steps and POC values
    time_steps = sim_data['All calculations (b:tnow in s)']
    poc_values = sim_data['POC (m:u1 in p.u.)']
    
    # Initialize the ideal results DataFrame
    ideal_results = pd.DataFrame({'All calculations (b:tnow in s) (Ideal)': time_steps, 'POC (m:u1 in p.u.) (Ideal)': np.nan})

    # Detect the drop time in the original simulation data
    drop_time_index = detect_drop_time(poc_values)
    
    if drop_time_index is not None:
        drop_time = time_steps.iloc[drop_time_index]
        print(f"Drop detected at time {drop_time} s")
        
        # Determine the steady state condition and set the final value accordingly
        last_half_avg = poc_values[int(len(poc_values) * 0.5):].mean()
        print(f"Average of the last half of the POC values: {last_half_avg}")
        
        
        # Set the POC value just before the drop to 1
        ideal_results.loc[ideal_results['All calculations (b:tnow in s) (Ideal)'] <= drop_time, 'POC (m:u1 in p.u.) (Ideal)'] = 1
        
        # here I am finding the min value so I can filter out all data before the min value and give that to the steady state function as if I dont do this it will see the time before voltage drop as steady state
        # also I need the min value to see if its within range and the ideal results need to be adjusted to match the actual fall rate
        # Step 1: Filtering
        filtered_poc_values = sim_data[(sim_data['POC (m:u1 in p.u.)'] < 1) & (sim_data['POC (m:u1 in p.u.)'] <= last_half_avg)]

        # Step 2: Finding the Minimum
        min_value = filtered_poc_values['POC (m:u1 in p.u.)'].min()
        print(f"Minimum value in the filtered POC data: {min_value}")

        # Assuming 'min_value' is the minimum POC value after which you want to consider data
        min_value_index = sim_data[sim_data['POC (m:u1 in p.u.)'] == min_value].index[0]
        min_value_time = time_steps.iloc[min_value_index]
        filtered_poc_values = sim_data.loc[min_value_index:]

        # To print the first 1000 rows of the DataFrame for debugging purposes
        print(filtered_poc_values['POC (m:u1 in p.u.)'].head(1000))


        # Detect the steady state in the original simulation data
        steady_state_index = detect_steady_state(filtered_poc_values['POC (m:u1 in p.u.)'], last_half_avg)
        print('Steady state index:')
        print(steady_state_index)
        
        if 0.92 < min_value < 1 :
            # here I am adjusting the fall rate of the ideal results to match the actual fall rate, so I dont badly score a good result
                    # Calculate the absolute difference from drop_time + 0.1 for all time steps
            time_differences = abs(ideal_results['All calculations (b:tnow in s) (Ideal)'] - min_value_time)

            # Find the index of the minimum difference
            closest_index = time_differences.idxmin()
            for i, value in enumerate(poc_values):
                if min_value < value < 0.98:
                    break

            first_interpolate_value = ideal_results.at[i, 'All calculations (b:tnow in s) (Ideal)']

            # Set the value 0.94 at the closest time step
            ideal_results.at[min_value_index, 'POC (m:u1 in p.u.) (Ideal)'] = min_value

            voltage_descent = ideal_results[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= first_interpolate_value].index.min()

            # Step 2: Set values to NaN within this range (excluding the start and end points)
            ideal_results.loc[(ideal_results.index > voltage_descent) & (ideal_results.index < min_value_index), 'POC (m:u1 in p.u.) (Ideal)'] = np.nan

            # Step 3: Perform linear interpolation
            ideal_results['POC (m:u1 in p.u.) (Ideal)'] = ideal_results['POC (m:u1 in p.u.) (Ideal)'].interpolate(method='linear')
            
        else:
                    # Calculate the absolute difference from drop_time + 0.1 for all time steps
            time_differences = abs(ideal_results['All calculations (b:tnow in s) (Ideal)'] - (drop_time + 0.1))

            # Find the index of the minimum difference
            closest_index = time_differences.idxmin()

            # Set the value 0.94 at the closest time step
            ideal_results.at[closest_index, 'POC (m:u1 in p.u.) (Ideal)'] = 0.94

            voltage_descent = ideal_results[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= drop_time].index.min()

            # Step 2: Set values to NaN within this range (excluding the start and end points)
            ideal_results.loc[(ideal_results.index > voltage_descent) & (ideal_results.index < closest_index), 'POC (m:u1 in p.u.) (Ideal)'] = np.nan

            # Step 3: Perform linear interpolation
            ideal_results['POC (m:u1 in p.u.) (Ideal)'] = ideal_results['POC (m:u1 in p.u.) (Ideal)'].interpolate(method='linear')


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
                ideal_results.loc[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= post_final_drop_time, 'POC (m:u1 in p.u.) (Ideal)'] = final_value
                """
                # Calculate the new slope for the ideal results to match the actual rise rate
                slope = (0.94 - 1) / actual_rise_time  # The drop is to 0.94 as per user's script
            
                # Apply this new slope to the ideal results
                for i in range(drop_time_index + 1, steady_state_index + 1):
                    time_diff = time_steps.iloc[i] - drop_time
                    ideal_results.at[i, 'POC (m:u1 in p.u.) (Ideal)'] = 1 + slope * time_diff

                # Re-interpolate if necessary for any subsequent points
                ideal_results['POC (m:u1 in p.u.) (Ideal)'] = ideal_results['POC (m:u1 in p.u.) (Ideal)'].interpolate(method='linear')
                """
                            # Step 1: Identify the range for interpolation
                # Find the index for post_final_drop_time
                post_final_drop_index = ideal_results[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= post_final_drop_time].index.min()

                # Step 2: Set values to NaN within this range (excluding the start and end points)
                ideal_results.loc[(ideal_results.index > closest_index) & (ideal_results.index < post_final_drop_index), 'POC (m:u1 in p.u.) (Ideal)'] = np.nan

                # Step 3: Perform linear interpolation
                ideal_results['POC (m:u1 in p.u.) (Ideal)'] = ideal_results['POC (m:u1 in p.u.) (Ideal)'].interpolate(method='linear')
                print("Ideal results adjusted to match the actual rise rate")
            else:
                print("Steady state not reached within the simulation time.")
                is_steady_state = 0.95 <= last_half_avg <= 1
                final_value = last_half_avg if is_steady_state else 1
                post_final_drop_time = drop_time + 0.1 + 0.5
                ideal_results.loc[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= post_final_drop_time, 'POC (m:u1 in p.u.) (Ideal)'] = final_value
     
                # Handle the case where the steady state is not found
                # Step 1: Identify the range for interpolation
                # Find the index for post_final_drop_time
                post_final_drop_index = ideal_results[ideal_results['All calculations (b:tnow in s) (Ideal)'] >= post_final_drop_time].index.min()

                # Step 2: Set values to NaN within this range (excluding the start and end points)
                ideal_results.loc[(ideal_results.index > closest_index) & (ideal_results.index < post_final_drop_index), 'POC (m:u1 in p.u.) (Ideal)'] = np.nan

                # Step 3: Perform linear interpolation
                ideal_results['POC (m:u1 in p.u.) (Ideal)'] = ideal_results['POC (m:u1 in p.u.) (Ideal)'].interpolate(method='linear')
                print("Ideal results were not adjusted to match the actual rise rate")

    else:
        print("No significant drop detected in the POC voltage")
        print("Likely Power Factory Error")

    # Save the final ideal results to an Excel file, left as a comment as it is not needed for the script but can be used for debugging !!!!!!!!!!!
    ideal_results_file_path = (r'C:\Users\JamesThornton\source\repos\Python Dissertation Script\Python Dissertation Script\Ideal Excel Results Voltage.csv') 
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
    filtered_sim_data = sim_data[sim_data['POC (m:u1 in p.u.)'] >= x_start]
    filtered_ideal_data = Ideal_Results_Table[Ideal_Results_Table['POC (m:u1 in p.u.) (Ideal)'] >= x_start]

    # Assuming the time and voltage columns are still correctly aligned after filtering
    #mae_time = np.abs(filtered_sim_data['All calculations (b:tnow in s)'] - filtered_ideal_data['All calculations (b:tnow in s) (Ideal)']).mean()
    mae_voltage = float(np.abs(filtered_sim_data['POC (m:u1 in p.u.)'] - filtered_ideal_data['POC (m:u1 in p.u.) (Ideal)']).mean())
    
    ############### Steady State Peak Ratio ###############
    
    # Load the Excel file

    # Select the column - replace 'ColumnName' with your actual column name
    column = sim_data['POC (m:u1 in p.u.)']

    # Find the lowest number in the column
    lowest_number = column.min()
    
    filtered_sim_data_max = sim_data[sim_data['All calculations (b:tnow in s)'] > lowest_number]
        
    # Find the maximum POC voltage value
    max_POC_voltage_value = filtered_sim_data_max['POC (m:u1 in p.u.)'].max()
    
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
    print("kp = " + str(kp))
    print("ki = " + str(ki))
    
    print("Fitness Value = " + str(fitness_value))
    
    matplotlib.pyplot.close('all')                                                          # Close all open figures to save memory 

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

    return (fitness_value,)
