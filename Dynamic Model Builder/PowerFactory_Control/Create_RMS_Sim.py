import PowerFactory_Control.Get_Nested_Folder as GNF

#RMS_Sim.Create_Simulation_Event(PF_Data, 'Voltage Drop', 2, '0.9', 'AC Voltage Source(3)')  -------- Example usage of the Create_Simulation_Event function

def Create_Simulation_Event(pf_data, Sim_Nam, Sim_Tim_Start, Vol__Drop_To, Sim_to_Control): 
    try:
        # Retrieve the simulation location from powerfactory
        pf_data.Simulation_Folder = GNF.get_nested_folder(pf_data, ['Study Cases', 'Power Flow', 'Simulation Events/Fault'])
        print(f"found folder: {pf_data.Simulation_Folder.loc_name}")  # Debug print

        Simulation_Object = pf_data.Simulation_Folder.CreateObject('EvtParam', Sim_Nam)
        print(f"Created simulation event: {Simulation_Object.loc_name}")  # Debug print

        Sim_to_Control_Object = pf_data.grid_folder.GetContents(Sim_to_Control)[0]
        print(f"Busbar object found: {Sim_to_Control_Object.loc_name}")  # Debug prin

        # Set attributes for the simulation event
        Simulation_Object.SetAttribute('e:time', float(Sim_Tim_Start))  
        Simulation_Object.SetAttribute('e:p_target', Sim_to_Control_Object)  # Set the time unit to seconds
        Simulation_Object.SetAttribute('e:variable', 'u0')  # For this to work you need to adjust the voltage source advanced RMS settings!
        Simulation_Object.SetAttribute('e:value', str(Vol__Drop_To))
         
        print(f"Simulation event created successfully: {Simulation_Object.loc_name}")  # Debug print

    except Exception as e:
        print(f"An error occurred: {e}")

def Create_Results_Variable(pf_data, Object_Monitor):
    #ry:
        # Retrieve the calculation location from powerfactory
        pf_data.Calculations_Folder = GNF.get_nested_folder(pf_data, ['Study Cases', 'Power Flow', 'All calculations'])
        print(f"found folder: {pf_data.Calculations_Folder.loc_name}")  # Debug print

        # Create the results variable in the specified folder
        Results_Object = pf_data.Calculations_Folder.CreateObject('IntMon', Object_Monitor)
        print(f"Created results variable: {Results_Object.loc_name}")  # Debug print

        # Set attributes for the results variable
        Object_Monitor_Object = pf_data.grid_folder.GetContents(Object_Monitor)[0]
        print(f"Busbar object found: {Object_Monitor_Object.loc_name}")  # Debug print

        # Set attributes for the results variable
        Results_Object.SetAttribute('e:obj_id', Object_Monitor_Object)  
        Results_Object.SetAttribute('e:vars', ['m:u'])  
        Results_Object.SetAttribute('e:vars', ['m:u1'])  # Set the time step for the results variable
        print(f"Results variable created successfully: {Results_Object.loc_name}")  # Debug print

   # except Exception as e:
    #    print(f"An error occurred: {e}")

def Add_Voltage_Source(pf_data, Busbar, voltage, R, X, U0, Isolated_System_TF):
    try:
        # Create the voltage source in the specified folder
        V_Source_Object = pf_data.grid_folder.CreateObject('ElmVac', Busbar + 'V_Source')
        print(f"Created voltage source: {V_Source_Object.loc_name}")  # Debug print

        # Create Cubicle for the voltage source
        Busbar_Object = pf_data.grid_folder.GetContents(Busbar)[0]
        BB_Cub = Busbar_Object.CreateObject('StaCubic', Busbar + '_Cub')  # Set the busbar for the voltage source
        print(f"Created cubicle for voltage source: {BB_Cub.loc_name}")  # Debug print

        if Isolated_System_TF == True:
            R = 0.0  # Set resistance to zero if the system is isolated
            X= 0.01  # Set reactance to a small value if the system is isolated
        elif Isolated_System_TF == False:
            U0 = 1

        # Set attributes for the voltage source
        V_Source_Object.SetAttribute('e:bus1', BB_Cub)  # Set the busbar for the voltage source
        V_Source_Object.SetAttribute('e:usetp', U0)  # Set the busbar for the voltage source
        V_Source_Object.SetAttribute('e:itype', 0)  # Set the type of voltage source
        V_Source_Object.SetAttribute('e:Unom', float(voltage))  # Set the nominal voltage level
        V_Source_Object.SetAttribute('e:R1', float(R))  # Set the Resistance level
        V_Source_Object.SetAttribute('e:X1', float(X))  # Set the Reactance level
        V_Source_Object.SetAttribute('e:leadUinput', 1)  # Set the voltage call
        V_Source_Object.SetAttribute('e:leadFinput', 1)  # Set the frequency call
        print(f"Voltage source populated successfully: {V_Source_Object.loc_name}")  # Debug print

    except Exception as e:
        print(f"An error occurred: {e}")