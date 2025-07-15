# pf_rms_toolkit_minimal.py  – depends only on pf_data
# --------------------------------------------------------------------
from __future__ import annotations
import datetime
from dataclasses import dataclass, field
from pathlib import Path
import PowerFactory_Control.Get_Nested_Folder as GNF

# ───────── option holders ────────────────────────────────────────────


@dataclass
class SimulationOptions:
    start_time: float = -0.1
    stop_time:  float = 20.0
    step_size:  float = 0.01
    ignore_dsl_warnings: bool = True


@dataclass
class ExportOptions:
    csv_path:  Path | str | None = None
    xlsx_path: Path | str | None = None


@dataclass
class MonitorSpec:
    channels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    use_existing: bool = False


# ────────── main class (NO direct pf imports / hints) ────────────────
class RMSRunner:
    """All PF access goes through the pf_data wrapper you pass in."""

    def __init__(self, pf_data):
        self.pf_data = pf_data                           # stash it

        # Ensure pf_data.project exists – otherwise derive it once
        if not hasattr(self.pf_data, "project"):
            self.pf_data.project = (
                self.pf_data.app.GetCurrentUser()
                .GetContents(f'{self.pf_data.project_name}.IntPrj')[0]
            )

    # ────────────────── helpers ──────────────────────────────────────
    def _get_or_make_monitor(self, study_case, element, var_codes):
        m_name = f"MON_{element.loc_name}"
        hits = study_case.GetContents(m_name + '.IntMon')
        mon  = hits[0] if hits else study_case.CreateObject('IntMon', m_name)

        mon.pElms = [element]
        # one line list-of-lists  [[elm,'u'],[elm,'i1'], … ]
        mon.cvars = [[element, v] for v in var_codes]
        return mon

    def _prepare_comsim(self, study_case, sim: SimulationOptions):
        comsim = study_case.GetContents('ComSim')
        comsim = comsim[0] if comsim else study_case.CreateObject('ComSim', 'Run Simulation')

        comsim.tstart      = sim.start_time
        comsim.tstop       = sim.stop_time
        comsim.dtgrd       = sim.step_size
        comsim.iopt_warn   = 0 if sim.ignore_dsl_warnings else 1
        return comsim

    def _export(self, study_case, elmres, exp: ExportOptions):
        """
        Export *elmres* to CSV (always) and XLSX (if requested).
        Prints a short notice when the file is written.
        """
        def try_export(ext: str, path: Path | None, code: int):
            if path is None:
                return

            comres = study_case.CreateObject("ComRes", f"Export_{ext}")
            comres.SetAttribute("pResult",    elmres)
            comres.SetAttribute("f_name",     str(path))
            comres.SetAttribute("iopt_exp",   code)   # 6 = CSV, 8 = XLSX
            comres.SetAttribute("iopt_locn",  1)      # English decimals
            comres.SetAttribute("ciopt_head", 1)      # header row

            rc = comres.Execute()
            comres.Delete()

            if rc == 0 and path.exists():
                print(f"    ↳ wrote {path.name}")
            else:
                print(f"    ⚠ export {ext} failed (code {rc})")

        try_export(".csv",  Path(exp.csv_path).with_suffix(".csv") if exp.csv_path else None, 6)
        try_export(".xlsx", Path(exp.xlsx_path).with_suffix(".xlsx") if exp.xlsx_path else None, 8)


    def run(
        self,
        *,
        study_case: str,
        monitors: MonitorSpec | None = None,
        sim: SimulationOptions       = SimulationOptions(),
        export: ExportOptions        = ExportOptions(),
    ):
        # ---------- locate & activate study case -------------------------------
        sc_folder  = GNF.get_nested_folder(self.pf_data, ["Study Cases"])
        sc_object  = sc_folder.GetContents(study_case)[0]
        sc_object.Activate()

        # ---------- optional monitors -----------------------------------------
        if monitors and monitors.channels:
            for elm_path, codes in monitors.channels.items():
                elm = self.pf_data.app.GetCalcRelevantObjects(elm_path)[0]
                self._get_or_make_monitor(sc_object, elm, codes)

        # ---------- prepare & execute ComSim ----------------------------------
        comsim = self._prepare_comsim(sc_object, sim)
        print(f"\n▶  Running ComSim in «{study_case}» "
              f"[{sim.start_time} … {sim.stop_time}s @ {sim.step_size}s]")
        rc = comsim.Execute()

        if rc != 0:
            raise RuntimeError(f"ComSim returned {rc} – simulation failed")
        print("✓  simulation finished")

        # ---------- quick stats ------------------------------------------------
        elmres = self.pf_data.app.GetFromStudyCase("All calculations.ElmRes")
    
        print(f"    result file: {elmres} variable(s)")

        # ---------- export -----------------------------------------------------
        if export.csv_path or export.xlsx_path:
            self._export(sc_object, elmres, export)


# ────────── helper for one-liner use  ────────────────────────────────
def quick_rms_run(
        pf_data,
        study_case: str,
        bus_name: str,
        out_dir: str,
        monitor_dict: dict[str, tuple[str, ...]] | None = None):

    #bus_obj  = pf_data.app.GetCalcRelevantObjects(f"{bus_name}.ElmTerm")[0]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    runner = RMSRunner(pf_data)
    runner.run(
        
        study_case = study_case,
        monitors   = MonitorSpec(monitor_dict or {}),
        export     = ExportOptions(
            csv_path = out_path / bus_name          #  ← no “.csv” here
            # xlsx_path = out_path / bus_name       #  add if you want XLSX
        )
    )


