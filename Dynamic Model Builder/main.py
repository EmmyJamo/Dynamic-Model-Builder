import DataClasses.Global_PF_Data as pf_data
import PowerFactory_Control.PowerFactory_SetUp as PF_Setup
import PowerFactory_Interaction.Generator_Gather as Gen
import PowerFactory_Interaction.Identify_Gens_of_Interest as GenIden
import PowerFactory_Interaction.Thevian_Equivalent as Thevian
import PowerFactory_Interaction.Gen_Isolation as GI
import PowerFactory_Interaction.Run_Initial_RMS as Run_In_RMS
import Data_Processing.Score_First_Data_Set as Score
import PowerFactory_Interaction.Tune_Isolated_Gens_Wrapper as Tune
import Data_Processing.Seed_AVR as avs


# PF_Data struct instance
pf_data = pf_data.PowerFactoryData()

# Project name
pf_data.project_name = '39 Bus New England System'


# Power Factory Setup
PF_Setup.powerfactory_setup(pf_data)


'''
# Build Infinite Bus Islands 
GI.build_infinite_bus_islands(pf_data)
'''
# Wrapper for Tuniong Isolated Generators
Tune.tune_selected_generators(pf_data)

'''


# Identify Generators of Interest
GenIden.run_generator_impact(pf_data)

avs.build_seeds_from_snapshot(
            pf_data,
            dip_level=0.9,
            t_drop=2.0,
            t_rise_short=2.2,
            t_rise_long=7.0)



# Gather Generators
Gen.Gather_Gens(pf_data)

# Build Thevenin Equivalent for Generators of Interest
Thevian.add_bus_thevenin_to_snapshot(pf_data)

SCENARIOS = [
    # name      drop_lvl  drop_t  rise_t
    ('fast_dip', 0.9,      2.0,    2.2),   # 0.2‑s hold 
    ('slow_hold', 0.9,     2.0,    7.0),   # 5‑s hold    (2 → 7 s)
]

# Builds and runs first set of voltage step simulations
Run_In_RMS.build_simulation_and_run(pf_data, SCENARIOS) # issue here!!! the folders simulation and calculations are not being created!!
# Evaluate Voltage Simulation Results
Score.update_bus_fitness(pf_data)



avs.build_seeds_from_snapshot(
            pf_data,
            dip_level=0.9,
            t_drop=2.0,
            t_rise_short=2.2,
            t_rise_long=7.0)



'''



