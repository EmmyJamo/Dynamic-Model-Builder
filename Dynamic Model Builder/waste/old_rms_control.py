


    

    # sim_data = sim_data.dropna(axis=1, how='all')  # Drop columns with all NaN values
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        
    sim_data.drop(index=1).to_csv(csv_path, index=False)   # drop 2nd row, save back
    print("? Second row removed")

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
