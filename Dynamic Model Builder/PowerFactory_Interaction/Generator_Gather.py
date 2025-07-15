import json
import os

def Gather_Gens(pf_data):
    base_dir = r'C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder\Dynamic Model Builder\JSON_DB\Network_Snapshots'
    filename = f"{pf_data.project_name}_gen_snapshot.json"
    out_path = os.path.join(base_dir, filename)

    def has_avr(comp_obj):

        if comp_obj is None:
            return False, None

        # all controller blocks are ElmDsl → search for *avr*
        for blk in comp_obj.GetContents('*.ElmDsl'):   # list of DataObjects
            if 'avr' in blk.loc_name.lower():          # case-insensitive match
                print(f"Found AVR block: {blk.loc_name}")
                return True, blk.loc_name

        return False, None

    all_trf = pf_data.grid_folder.GetContents("*.ElmTr2")

    def find_transformer(gen_bus):
        """
        gen_bus   … IntBus object on which the generator cubicle sits
        returns   … (has_trf : bool,
                      trf_name : str | None,
                      grid_bus_name : str | None)
        """

        for trf in all_trf:

            print(f"Checking transformer: {trf.loc_name}")
            term = trf.GetAttribute("buslv")          # ElmTerm (cubicle)
            bus  = term.GetAttribute("cterm")        # IntBus  (busbar)
            lv_bus = bus.loc_name                     # <-- real bus name
            print(lv_bus)

            # --------------------- is THIS transformer connected to the gen? ----
            if lv_bus == gen_bus:
                termhv = trf.GetAttribute("bushv")          # ElmTerm (cubicle)
                bus_namehv  = termhv.GetAttribute("cterm")        # IntBus  (busbar)
                bus_namehv = bus_namehv.loc_name  
                print(bus_namehv)
                print(f'found transformer')

                return True, trf.loc_name, bus_namehv 
                
            else:
                # no transformer whose LV bus matches gen_bus
                print(f'did not find transformer')
        return False, None, None

    # --- force project-wide reactive-power units to MVar --------------------------
    settings_root   = pf_data.project.GetContents('Settings.*')[0]          # root
    proj_settings   = settings_root.GetContents('Project Settings.*')[0]    # SetPrj

    proj_settings.SetAttribute('cspqexp',    'M')   # Loads / Asyn / DC P,Q,S
    proj_settings.SetAttribute('cspqexpgen', 'M')   # Static & Synch Gen P,Q,S

    print("Project Settings updated: all Q units now displayed in MVar.")

    gens = []

    #  synchronous machines
    for sym in pf_data.app.GetCalcRelevantObjects("*.ElmSym"):
        typ = sym.GetAttribute("typ_id")
        try:
            term = sym.GetAttribute("bus1")          # ElmTerm (cubicle)
            bus  = term.GetAttribute("cterm")        # IntBus  (busbar)
            print(bus.loc_name)
            # Does it have an AVR
            Plant_Name = sym.GetAttribute("e:c_pmod")
            #print(f"Plant Name: {Plant_Name}")

            has_AVR, avr_name = has_avr(Plant_Name)

            trf_flag, trf_name, grid_bus = find_transformer(bus.loc_name)

            gens.append({
                "name":      sym.loc_name,
                "type":      "synchronous",
                "subtype":   sym.GetAttribute("e:cCategory"),
                "bus":       bus.loc_name,           # <-- real bus name
                "MVA_rated": typ.GetAttribute("e:sgn"),
                "Cubicle": term.loc_name,  # <-- cubicle name"
                "voltage": typ.GetAttribute("e:ugn"),  # <-- voltage level
                "MVar_Max": sym.GetAttribute("e:cQ_max"),
                "MVar_Min": sym.GetAttribute("e:cQ_min"),
                "AVR"     : has_AVR,  # True if AVR exists, False otherwise
                "AVR_Name": avr_name if has_AVR else None,  # Name of AVR block if exists"
                "Has_Trf":     trf_flag,
                "Trf_Name":    trf_name,
                "Grid_Bus":    grid_bus,
            })
            print(f"GenSym {sym.loc_name} Recorded")
        except KeyError:
            print(f"KeyError for synchronous generator {sym.loc_name}. Skipping...")

    # ---------- inverter-based generators ----------
    for vsc in pf_data.app.GetCalcRelevantObjects("*.ElmGenstat"):
        typ = vsc.GetAttribute("typ_id")             # <-- don’t reuse old ‘typ’
        try:
            term = vsc.GetAttribute("bus1")          # ElmTerm (cubicle)
            bus  = term.GetAttribute("cterm")        # IntBus  (busbar)

            trf_flag, trf_name, grid_bus = find_transformer(term)

            gens.append({
                "name":      vsc.loc_name,
                "type":      "inverter",
                "subtype":   vsc.GetAttribute("e:cCategory"),
                "bus":       bus.loc_name,           # <-- real bus name
                "MVA_rated": typ.GetAttribute("e:sgn"),
                "Cubicle": term.loc_name,  # <-- cubicle name"
                "voltage": typ.GetAttribute("e:ugn"),  # <-- voltage level
                "MVar_Max": vsc.GetAttribute("e:cQ_max"),
                "MVar_Min": vsc.GetAttribute("e:cQ_min"),
                
                "Has_Trf":     trf_flag,
                "Trf_Name":    trf_name,
                "Grid_Bus":    grid_bus,
            })
        except KeyError:
            print(f"KeyError for inverter {vsc.loc_name}. Skipping...")

    # ---------- write JSON ----------
    data = {"generators": gens}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote detailed network snapshot (no KeyError) to:\n    {out_path}")
