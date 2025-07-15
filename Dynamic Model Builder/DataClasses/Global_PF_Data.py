from dataclasses import dataclass


@dataclass
class PowerFactoryData:
    def __init__(self):
        self.app                    = None
        self.project_name           = None
        self.project_entry          = None
        self.op_scen_name           = 'Normal'
        self.poc_busbar_name        = 'POC.ElmTerm'
        self.dsl_name               = 'Custom Plant Control.ElmDsl'
        self.user                   = None
        self.projects               = None
        self.op_scenario_folder     = None
        self.op_scen                = None
        self.grid_folder            = None
        self.Calculations_Folder    = None
        self.Simulation_Folder      = None
        self.variations_folder      = None

