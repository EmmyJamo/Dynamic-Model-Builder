
class VoltageStepConfig:
    """
    All numeric inputs for the dip / swell.
    Instantiate → tweak → pass to VoltageStepBuilder.
    """
    def __init__(self):
        self.t_start: float = 0.0,
        self.t_stop: float = 15.0,
        self.time_step: float = 0.01,
        self.dip_pu: float = 0.90,
        self.dip_t0: float = 1.0,
        self.rec_t0: float = 8.0

class CaseMeta:
    """
    Non-numeric meta options (folder names, flags, variant pointers, …)
    """
    def __init__(self):
        self.case_name = "VoltageStep_SC",
        self.variant=None,                 # IntScheme  (None → active)
        self.expansion=None,               # IntSstage  (None → default)
        self.overwrite_monitors: bool = True,
        self.verbose: bool = True