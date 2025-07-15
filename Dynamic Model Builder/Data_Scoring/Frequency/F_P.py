from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Data_Scoring.Attribute_Detection.Functions import detect_drop_time, detect_steady_state, custom_log_interpolation, detect_oscillations, calculate_log_values
import matplotlib

def evaluate_frequency_control(individual, pf_data, return_gradients=False):
    
    ############## Setting up Power Factory Parameters ##############

    poc_busbar = pf_data.app.GetCalcRelevantObjects(pf_data.poc_busbar_name)[0]
    control_dsl = pf_data.app.GetCalcRelevantObjects(pf_data.dsl_name)[0]
    ki, kp = individual                                                                          # list of ki values to iterate, the values are created in the genetic algorithm below 
         
    ############## Setting ki and kp values in the simulation  ##############

    params = control_dsl.params                                                             # Set up parameters for Power Factory
    params[20] = kp.item() if isinstance(kp, np.ndarray) and kp.size == 1 else float(kp)
    params[21] = ki.item() if isinstance(ki, np.ndarray) and ki.size == 1 else float(ki)
    #params[20] = float(kp)                                                                        # set kp (proportional) value in Power Factory
    #params[21] = float(ki)                                                                           # set ki (integral) value in Power Factory        
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
    poc_values = sim_data['POC SW (m:P:bus2 in MW)']
    
    # Initialize the ideal results DataFrame
    ideal_results = pd.DataFrame({'All calculations (b:tnow in s) (Ideal)': time_steps, 'POC SW (m:P:bus2 in MW) (Ideal)': np.nan})

    # Detect the drop time in the original simulation data
    drop_time_index = detect_drop_time(poc_values)
    
    if drop_time_index is not None:
        drop_time = time_steps.iloc[drop_time_index]
        print(f"Drop detected at time {drop_time} s")
        
        # Determine the steady state condition and set the final value accordingly
        last_half_avg = poc_values[int(len(poc_values) * 0.2):].mean()
        print(f"Average of the last half of the POC values: {last_half_avg}")
        
        # find real power before frequency rise
        real_power_before = sim_data['POC SW (m:P:bus2 in MW)'].iloc[drop_time_index-1]


        # Set the POC value just before the drop to 1
        ideal_results.loc[ideal_results['All calculations (b:tnow in s) (Ideal)'] <= drop_time, 'POC SW (m:P:bus2 in MW) (Ideal)'] = real_power_before

        filtered_poc_values = sim_data.loc[drop_time_index + 2000:]

        # To print the first 1000 rows of the DataFrame for debugging purposes
        print(filtered_poc_values['POC (m:u1 in p.u.)'].head(1000))


        # Detect the steady state in the original simulation data
        steady_state_index = detect_steady_state(filtered_poc_values['POC SW (m:P:bus2 in MW)'], last_half_avg)
        print('Steady state index:')
        print(steady_state_index)    
        

        if steady_state_index is not None:
            # Adjust the steady_state_index with the starting index of filtered_poc_values
            adjusted_index = steady_state_index + drop_time_index + 2000
            steady_state_time = time_steps.iloc[adjusted_index]
            print(f"Steady state reached at time {steady_state_time} s")
            actual_fall_time = time_steps.iloc[adjusted_index] - drop_time
            print(f"Actual fall time: {actual_fall_time} s")


            # Calculate the parameters for the log function
            a, b, c = custom_log_interpolation(drop_time, real_power_before, steady_state_time, last_half_avg)

            # Now, create the time array for which you want to calculate the log function values
            # This should be the time points between drop_time and steady_state_time
            interpolation_time_array = time_steps[(time_steps >= drop_time) & (time_steps <= steady_state_time)]

            # Calculate the interpolated values using the parameters
            interpolated_values = calculate_log_values(interpolation_time_array, a, b, c, drop_time)


            # Apply the interpolated values to the 'ideal_results' DataFrame
            ideal_results.loc[(ideal_results['All calculations (b:tnow in s) (Ideal)'] >= drop_time) & 
                              (ideal_results['All calculations (b:tnow in s) (Ideal)'] <= steady_state_time),
                              'POC SW (m:P:bus2 in MW) (Ideal)'] = interpolated_values
            # Set all values after the steady state point to the steady state value
            ideal_results.loc[ideal_results['All calculations (b:tnow in s) (Ideal)'] > steady_state_time,
                              'POC SW (m:P:bus2 in MW) (Ideal)'] = last_half_avg
        else :
            print("steady state not reached")
            

    # Save the final ideal results to an Excel file, left as a comment as it is not needed for the script but can be used for debugging !!!!!!!!!!!
    ideal_results_file_path = (r'C:\Users\JamesThornton\source\repos\Python Dissertation Script\Python Dissertation Script\Ideal Excel Results Frequency.csv') 
    ideal_results.to_csv(ideal_results_file_path, index=False)
    
    ############### Importing Ideal Data ###############

    Ideal_Results_Table = ideal_results             #pd.read_csv(r'C:\Users\JamesThornton\source\repos\Python Dissertation Script\Python Dissertation Script\Ideal Excel Results.csv')
            


    oscillation_dictionary = detect_oscillations(sim_data, 'POC SW (m:P:bus2 in MW)')
    print(f"Oscillation Count: {oscillation_dictionary['count']}")
    print(f"Oscillation Amplitudes: {oscillation_dictionary['amplitudes']}")
    print(f"Oscillation Frequency: {oscillation_dictionary['frequency']}")
    print(f"Oscillation Duration: {oscillation_dictionary['duration']}")


    x_start = 0
        
    # Filter the data
    filtered_sim_data = sim_data[sim_data['POC SW (m:P:bus2 in MW)'] >= x_start]
    filtered_ideal_data = Ideal_Results_Table[Ideal_Results_Table['POC SW (m:P:bus2 in MW) (Ideal)'] >= x_start]

    # Assuming the time and voltage columns are still correctly aligned after filtering
    #mae_time = np.abs(filtered_sim_data['All calculations (b:tnow in s)'] - filtered_ideal_data['All calculations (b:tnow in s) (Ideal)']).mean()
    mae_voltage = float(np.abs(filtered_sim_data['POC SW (m:P:bus2 in MW)'] - filtered_ideal_data['POC SW (m:P:bus2 in MW) (Ideal)']).mean())


    #print("Time column Mean Absolute Error (after x_start) = " + str(mae_time))
    print("Voltage column Mean Absolute Error (after x_start) = " + str(mae_voltage))
    #print("Max POC Voltage Value = " + str(max_POC_voltage_value))
    print("kp = " + str(kp))
    print("ki = " + str(ki))
    
    print("Fitness Value = " + str(mae_voltage))
    
    matplotlib.pyplot.close('all')                                                          # Close all open figures to save memory 

    if return_gradients:
        # Approximate the gradient with respect to kp
        delta = 0.01  # Small change to apply to kp and ki
        kp_perturbed = kp + delta
        fitness_perturbed_kp = evaluate_frequency_control((ki, kp_perturbed), pf_data, return_gradients=False)  # Evaluate fitness with perturbed kp without gradients
        grad_kp = (fitness_perturbed_kp - mae_voltage) / delta  # Approximate partial derivative with respect to kp

        # Approximate the gradient with respect to ki
        ki_perturbed = ki + delta
        fitness_perturbed_ki = evaluate_frequency_control((ki_perturbed, kp), pf_data, return_gradients=False)  # Evaluate fitness with perturbed ki without gradients
        grad_ki = (fitness_perturbed_ki - mae_voltage) / delta  # Approximate partial derivative with respect to ki

        return mae_voltage, grad_kp, grad_ki

    return (mae_voltage,)