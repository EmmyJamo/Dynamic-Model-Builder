import json
import numpy as np

def snapshot_network(pf_data):
    """
    Reads all generators and network topology from the active PF study case,
    builds a positive-sequence bus Z matrix including:
      - Lines (scaled R/X from their typ_id by actual length)
      - Two‐winding & three‐winding transformers (R/X from typ_id)
    and writes the data to JSON.
    """

    app     = pf_data.app
    project = pf_data.project
    name    = pf_data.project_name
    out_path = (
        r'C:\Users\JamesThornton\source\repos\Dynamic Model Builder'
        r'\Dynamic Model Builder\JSON_DB\Network_Snapshots'
        + "\\" + name + "_network_snapshot.json"
    )

    # --- 1) Gather all bus objects referenced anywhere ---
    bus_set = {}

    def collect_bus(bus_obj):
        """Add bus to dict by name."""
        bus_set[bus_obj.loc_name] = bus_obj

    # Lines
    for line in app.GetCalcRelevantObjects("*.ElmLne"):
        collect_bus(line.GetAttribute("e:bus1"))
        collect_bus(line.GetAttribute("e:bus2"))

    # 2-winding transformers
    for tr in app.GetCalcRelevantObjects("*.ElmTr2"):
        collect_bus(tr.GetAttribute("e:bushv"))
        collect_bus(tr.GetAttribute("e:buslv"))
    '''
    # 3-winding transformers
    for tr3 in app.GetCalcRelevantObjects("*.ElmTr3"):
        collect_bus(tr3.GetAttribute("bus1"))
        collect_bus(tr3.GetAttribute("bus2"))
        collect_bus(tr3.GetAttribute("bus3"))
    '''
    # Generators
    for sym in app.GetCalcRelevantObjects("*.ElmSym"):
        collect_bus(sym.GetAttribute("e:bus1"))
    for vsc in app.GetCalcRelevantObjects("*.ElmGenstat"):
        collect_bus(vsc.GetAttribute("e:bus1"))

    # Final bus list and index
    bus_names = list(bus_set.keys())
    idx       = {name: i for i, name in enumerate(bus_names)}
    n         = len(bus_names)

    # --- 2) Initialize Z matrix ---
    Z = np.zeros((n, n), dtype=complex)

    # --- 3) Branch helper ---
    def add_branch(bA, bB, R, X):
        i, j = idx[bA.loc_name], idx[bB.loc_name]
        z = complex(R, X)
        Z[i, j] += z
        Z[j, i] += z

    # --- 4) Lines: scale R/X from typ_id by length ---
    for line in app.GetCalcRelevantObjects("*.ElmLne"):
        typ       = line.GetAttribute("typ_id")    # ElmTypLne
        length_km = line.GetAttribute("e:dline")   # km
        R1        = typ.GetAttribute("e:rline")    # Ω/km
        X1        = typ.GetAttribute("e:xline")    # Ω/km

        R = R1 * length_km
        X = X1 * length_km
        b1 = line.GetAttribute("e:bus1")
        b2 = line.GetAttribute("e:bus2")
        add_branch(b1, b2, R, X)

    # --- 5) Two-winding transformers via typ_id ---
    # ensure PF is set to per-unit Rx mode
    settings      = project.GetContents('Settings.*')[0]
    input_options = settings.CreateObject('IntOpt', 'Input Options')
    tr2_settings  = input_options.CreateObject('OptTyptr2', '2W Tr Settings')
    tr2_settings.SetAttribute('iopt_uk', 'rx')

    for tr in app.GetCalcRelevantObjects("*.ElmTr2"):
        typ = tr.GetAttribute("typ_id")        # ElmTypTr2
        R   = typ.GetAttribute("e:r1pu")
        X   = typ.GetAttribute("e:x1pu")
        b1  = tr.GetAttribute("e:bushv")
        b2  = tr.GetAttribute("e:buslv")
        add_branch(b1, b2, R, X)
    '''
    # --- 6) Three-winding transformers via typ_id ---
    for tr3 in app.GetCalcRelevantObjects("*.ElmTr3"):
        typ       = tr3.GetAttribute("typ_id")      # ElmTypTr3
        R12, X12  = typ.GetAttribute("e:r12"), typ.GetAttribute("e:x12")
        R23, X23  = typ.GetAttribute("e:r23"), typ.GetAttribute("e:x23")
        R31, X31  = typ.GetAttribute("e:r31"), typ.GetAttribute("e:x31")
        b1, b2, b3 = (tr3.GetAttribute(a) for a in ("bus1","bus2","bus3"))
        add_branch(b1, b2, R12, X12)
        add_branch(b2, b3, R23, X23)
        add_branch(b3, b1, R31, X31)
    '''
    # --- 7) Generators with typ_id data ---
    gens = []
    for sym in app.GetCalcRelevantObjects("*.ElmSym"):
        typ = sym.GetAttribute("typ_id")
        gens.append({
            "name":      sym.loc_name,
            "type":      "synchronous",
            "subtype":   sym.GetAttribute("e:cCategory"),
            "bus":       sym.GetAttribute("e:bus1").loc_name,
            "MVA_rated": typ.GetAttribute("e:sgn"),
        })
    for vsc in app.GetCalcRelevantObjects("*.ElmGenstat"):
        gens.append({
            "name":      vsc.loc_name,
            "type":      "inverter",
            "subtype":   vsc.GetAttribute("e:cCategory"),
            "bus":       vsc.GetAttribute("e:bus1").loc_name,
            "MVA_rated": typ.GetAttribute("e:sgn"),
        })

    # --- 8) Write out JSON ---
    data = {
        "buses":       bus_names,
        "Z":           [[z.real, z.imag] for z in Z.flatten()],
        "generators":  gens
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote detailed network snapshot (no KeyError) to:\n    {out_path}")
