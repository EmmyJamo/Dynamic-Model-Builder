# ---------------------------------------------------------------------------
# pf_thevenin_tools.py
#   –  Positive-sequence Thévenin extraction helpers
#     * add_thevenin_to_snapshot(...)  – existing: for generators-of-interest
#     * add_bus_thevenin_to_snapshot(...) – NEW: full system bus list
#
# Works in PowerFactory 2022–2024, Python ≥3.8
# ---------------------------------------------------------------------------
from pathlib import Path
import json
from datetime import datetime

# ───────────────────────────────────────────────────────────────
# configuration
# ───────────────────────────────────────────────────────────────
_JSON_BASE = (
    r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
    r"\Dynamic Model Builder\JSON_DB\Network_Snapshots"
)

def _snapshot_path(pf_data) -> Path:
    return Path(_JSON_BASE) / f"{pf_data.project_name}_gen_snapshot.json"


# ───────────────────────────────────────────────────────────────
# low-level impedance routine (re-used by both helpers)
# ───────────────────────────────────────────────────────────────
def _run_shc_get_z(app, bus_obj):
    """
    IEC-60909 3-phase bolted fault at *bus_obj*.
    Returns (Rth, Xth, phiui, X/R, R/X, X/R_base, R/X_base, Skss, Ikss, ip).
    Raises RuntimeError if ComShc fails.
    """
    shc = app.GetFromStudyCase("ComShc")
    if shc is None:
        raise RuntimeError("No ComShc object in the active study case")

    shc.iopt_mde     = 0          # IEC bolted 3-phase
    shc.iopt_sim     = 0          # AC component only
    shc.p_faultobj   = bus_obj
    shc.frmLimitsBrc = 0

    if shc.Execute() != 0:
        raise RuntimeError(f"Short-circuit calc failed at {bus_obj.loc_name}")

    # PF exposes the Thevenin quantities directly on the faulted bus
    return tuple(float(bus_obj.GetAttribute(tag)) for tag in (
        "m:R", "m:X", "m:phiui",
        "m:XtoR", "m:RtoX",
        "m:XtoR_b", "m:RtoX_b",
        "m:Skss", "m:Ikss", "m:ip")
    )

'''
# ───────────────────────────────────────────────────────────────
# 1)  Existing function – unchanged
# ───────────────────────────────────────────────────────────────
def add_thevenin_to_snapshot(pf_data):
    """
    Append Rth / Xth to each generator that already carries
    `"selected_for_tuning": true` in the snapshot file.
    """
    js_path = _snapshot_path(pf_data)
    if not js_path.exists():
        raise FileNotFoundError(js_path)

    snap = json.loads(js_path.read_text())

    # ---- name → ElmTerm LUT -------------------------------------------------
    all_terms   = pf_data.app.GetCalcRelevantObjects("*.ElmTerm")
    term_by_nm  = {t.loc_name: t for t in all_terms}

    for g in snap.get("generators", []):
        if not g.get("selected_for_tuning"):
            continue

        term = term_by_nm.get(g["bus"])
        if term is None:
            print(f"⚠️  Bus '{g['bus']}' not found – skipping {g['name']}")
            continue

        try:
            (R,X,phiui,XtoR,RtoX,XtoRb,RtoXb,Skss,Ikss,ip) = _run_shc_get_z(pf_data.app, term)
            g.update({
                "Rth_ohm"   : R,
                "Xth_ohm"   : X,
                "phiui"     : phiui,
                "XtoR"      : XtoR,
                "RtoX"      : RtoX,
                "XtoR_b"    : XtoRb,
                "RtoX_b"    : RtoXb,
                "Skss_MVA"  : Skss,
                "Ikss_kA"   : Ikss,
                "ip_A"      : ip
            })
            print(f"✓ {g['name']:20s}  Zth = {R:.4f} + j{X:.4f} Ω")
        except Exception as err:
            print(f"⚠️  {g['name']}: {err}")

    snap["Thevenin_update_ts"] = datetime.now().isoformat(timespec="seconds")
    js_path.write_text(json.dumps(snap, indent=2))
    print(f"\n✅  Generator Thévenin data written to {js_path}")
'''

# ───────────────────────────────────────────────────────────────
# 2)  NEW – complete bus list
# ───────────────────────────────────────────────────────────────
def add_bus_thevenin_to_snapshot(pf_data):

    """
    For every ElmTerm in the active grid:

        • run a *single* balanced positive-sequence load-flow (ComLdf)
          so that m:u / m:u1 are available,
        • obtain U₀ (steady-state RMS voltage in pu) from the load-flow,
        • obtain Rth / Xth etc. from an IEC-60909 3-φ bolted fault,
        • write/overwrite the "buses" array in the snapshot JSON.

    Added JSON fields (per bus):
        "U0_pu"    : 0.9873            # steady-state voltage magnitude
        "V_nom_kV" : 110.0             # already added previously
    """
    js_path = _snapshot_path(pf_data)
    snap    = json.loads(js_path.read_text()) if js_path.exists() else {}

    app  = pf_data.app
    prj  = pf_data.project          # make sure the study case is active



    # ------------------------------------------------------------------
    # 2) iterate over all terminals, collect Zth + U0
    # ------------------------------------------------------------------
    bus_records = []
    terms = app.GetCalcRelevantObjects("*.ElmTerm")

    for term in terms:
        print(term)
        try:

            # ------------------------------------------------------------------
            # 1) run a load-flow once – all bus voltages become available
            # ------------------------------------------------------------------
            ldf = app.GetFromStudyCase("ComLdf")
            if ldf is None:
                raise RuntimeError("No ComLdf object in the study case")

            # optional: enforce balanced / positive-sequence settings here
            ldf.iopt_net = 0            # 0 = balanced, positive-sequence
            if ldf.Execute() != 0:
                raise RuntimeError("❌ Load-flow (ComLdf) failed – aborting")
            else:
                print("✓ Load-flow executed – steady-state voltages available")

            U0 = round(term.GetAttribute("m:u"),3) 
            Ukv = round(term.GetAttribute("m:Ul"),3) 

            

            (R, X, phiui, XtoR, RtoX,
             XtoRb, RtoXb, Skss, Ikss, ip) = _run_shc_get_z(app, term)

            V_nom = term.GetAttribute("uknom") or 0.0  # kV line-to-line

            bus_records.append({
                "name"      : term.loc_name,
                "V_nom_kV"  : V_nom,
                "U0_pu"     : U0,               # ← NEW
                "Rth_ohm"   : R,
                "Xth_ohm"   : X,
                "phiui_deg" : phiui,
                "XtoR"      : XtoR,
                "RtoX"      : RtoX,
                "XtoR_base" : XtoRb,
                "RtoX_base" : RtoXb,
                "Skss_MVA"  : Skss,
                "Ikss_kA"   : Ikss,
                "ip_A"      : ip
            })
            print(f"✓ {term.loc_name:25s}  U₀ = {U0:.4f} pu  "
                  f"Zth = {R:.4f} + j{X:.4f} Ω")
        except Exception as err:
            print(f"⚠️  {term.loc_name}: {err}")

    # ------------------------------------------------------------------
    # 3) write back
    # ------------------------------------------------------------------
    snap["buses"]                       = bus_records
    snap["Bus_Thevenin_update_ts"]      = datetime.now().isoformat(timespec="seconds")
    snap["Bus_Voltage_update_ts"]       = snap["Bus_Thevenin_update_ts"]

    with open(js_path, "w", encoding="utf-8") as fp:
        json.dump(snap, fp, indent=2)

    print(f"\n✅  Impedance + steady-state voltage for {len(bus_records)} buses "
          f"written to {js_path}")
