# voltage_step_case.py
# ---------------------------------------------------------------------------
# Build / update / run an RMS “voltage-step” study-case in PowerFactory
# Author: James Thornton (modular version)
# ---------------------------------------------------------------------------
from __future__ import annotations
import PowerFactory_Control.Get_Nested_Folder as GNF

# ╭──────────────────────────────────────────────────────────────────────╮
# │ 1.  LOGIC-CLASS  – builder object                                   │
# ╰──────────────────────────────────────────────────────────────────────╯
class VoltageStepBuilder:
    """Create / update / run a dip-and-recovery study case."""

    def __init__(self, pf_data):
        """
        Parameters
        ----------
        pf_data : a wrapper with .app, .project (IntPrj) already set,
                  plus any other helpers you store there (grid_folder…).
        """
        self.app = pf_data.app
        self.prj = pf_data.project

    # ───────────────────────────────────────────────────────── helpers ──
    def _resolve_variant(self, meta):
        if meta.variant:
            return meta.variant
        variants = self.prj.GetContents('*.IntScheme')
        active   = [v for v in variants if v.GetAttribute('is_act')]
        if not active:
            raise RuntimeError("No active variant found.")
        return active[0]

    def _study_case_folder(self, pf_data):
        # location:  Network Model / StudyCases.IntCase
        return GNF.get_nested_folder(pf_data, ['Study Cases.IntPrjfolder'])

    # ─────────────────────────────────────────────────── public API ─────
    def build_case(self, pf_data, Study_Case_Config, Study_Case_Meta):
        """
        Create or refresh the IntStudyCase «meta.case_name» according to *cfg*.
        Returns the IntStudyCase handle (ready to run).
        """
        sc_folder = self._study_case_folder(pf_data)

        if Study_Case_Meta.case_name in sc_folder.GetContents('*.IntCase'):
            print(f"➖  variant «{Study_Case_Meta.case_name}» already exists, using it")
            return sc_folder.GetContents(Study_Case_Meta.case_name)[0]
        else:
            print(f"➕  creating new study case «{Study_Case_Meta.case_name}»")
            sc = sc_folder.CreateObject('IntCase', Study_Case_Meta.case_name)
            sc.Activate()
            print(f"Var activated" + Study_Case_Meta.case_name)       



        '''
        # 1. variant / expansion
        variant = self._resolve_variant(Study_Case_Meta)
        sc.pVariant = variant
        if Study_Case_Meta.expansion:
            sc.pExpansion = Study_Case_Meta.expansion
        '''
        # 2. global RMS options
        sc.iopt_sim  = 0               # 0 = RMS
        sc.tstart    = Study_Case_Config.t_start
        sc.tstop     = Study_Case_Config.t_stop
        sc.Method    = 1               # fixed step
        sc.dtgrd     = Study_Case_Config.time_step
        sc.iopt_lf   = 1               # run LF initialisation

        # 3. wipe old events / monitors
        for kid in sc.GetChildren():
            kid.Delete()
        '''
        # 4. events: dip + recovery on every External Grid
        ext_grids = variant.GetContents('*.ElmXnet')
        for eg in ext_grids:
            for t0, val in ((Study_Case_Config.dip_t0, Study_Case_Config.dip_pu),
                            (Study_Case_Config.rec_t0, 1.0)):
                ev = sc.CreateObject('IntEvent',
                                     f"EV_{eg.loc_name}_{t0}")
                ev.time = t0
                ev.element = eg
                ev.var = 'uknom'
                ev.value = val

        # 5. monitors
        synms = variant.GetContents('*.ElmSym,*.ElmGenstat')
        if Study_Case_Meta.overwrite_monitors:
            for g in synms:
                mon = sc.CreateObject('IntMon', f"MON_{g.loc_name}")
                mon.pElms = [g]
                mon.iopt_object = 0     # element itself
                mon.iopt_allpq  = 0
                mon.iopt_aspen  = 1

        if Study_Case_Meta.verbose:
            print(f"   → {len(ext_grids)} grids dip to {Study_Case_Config.dip_pu} pu")
            print(f"   → {len(synms)} generator monitors added")
            print(f"✅  study case «{Study_Case_Meta.case_name}» ready")
        '''
        return sc

    # -------------------------------------------------------------------
    def run_case(self, sc):
        """Activate and run the given IntStudyCase."""
        sc.Activate()
        self.app.RunStudyCase(sc)
        print(f"🏁  study case «{sc.loc_name}» finished.")





