import json, os
from datetime import datetime

def rank_gens_to_json(pf_data,
                      dq_step=1,
                      metric="rms",
                      weight_headroom=True,
                      top_fraction=0.30):

    # ------------------------------------------------------------------ #
    # 1) open snapshot ................................................. #
    base_dir = (r"C:\Users\james\source\repos\EmmyJamo\Dynamic-Model-Builder"
                r"\Dynamic Model Builder\JSON_DB\Network_Snapshots")
    snap_path = os.path.join(base_dir,
                             f"{pf_data.project_name}_gen_snapshot.json")
    if not os.path.isfile(snap_path):
        raise FileNotFoundError("Snapshot not found: " + snap_path)

    with open(snap_path, "r", encoding="utf-8") as fp:
        snapshot = json.load(fp)
    gen_info_name = {g["name"]: g for g in snapshot["generators"]}
    #gen_info_type = {g["type"]: g for g in snapshot["generators"]}

    # ------------------------------------------------------------------ #
    # 2) PF handles .................................................... #
    app  = pf_data.app
    ldf  = app.GetFromStudyCase("ComLdf")
    grid = app.GetCalcRelevantObjects('*.ElmNet')[0]      # master grid folder
    buses = app.GetCalcRelevantObjects('*.ElmBus')

    if ldf.Execute() != 0:
        raise RuntimeError("Base load-flow did not converge.")
    v_base = {b.loc_name: b.GetAttribute("m:u") for b in buses}

    # map generator → its terminal bus (ElmTerm)
    name_to_bus = {}
    for gen in (app.GetCalcRelevantObjects("*.ElmSym") +
                app.GetCalcRelevantObjects("*.ElmGenstat")):
        term = gen.GetAttribute("bus1")
        if term:
            name_to_bus[gen.loc_name] = term

    scores = {}

    # ------------------------------------------------------------------ #
    # 3) finite-difference via grid-level shunt ........................ #
    for gname, gdict in gen_info_name.items():
        term = name_to_bus.get(gname)
        print({term})
        if term is None:
            print(" ! no bus for", gname, "– skipping")
            continue

        rated_kv = gdict.get("voltage")             # value you saved earlier
        if rated_kv is None:
            print("  ! no rated-voltage for", gname, "— skipping")
            continue

        genbus = gdict.get("bus")  + '.ElmTerm'  # bus name + 'ElmTerm' suffix'
        if genbus is None:
            print("  ! no rated-bus for", gname, "— skipping")
            continue

        print(f"gname={gname}, genbus={genbus}, rated_kv={rated_kv}, dq_step={dq_step}")

        '''
        # create shunt **in the Grid folder**, then attach to bus
        shunt = pf_data.project.CreateObject('ElmShnt', gname + '_temp_sh') 
        pf_data.grid_folder.Move(shunt)

        shunt_gf = pf_data.grid_folder.GetContents(gname + '_temp_sh')[0]

        Node_contents = pf_data.grid_folder.GetContents(genbus)

        Node = Node_contents[0]

        Node_Cubicle = Node.CreateObject('StaCubic', gname + 'C_Shu')

        shunt_gf.SetAttribute('e:bus1', Node_Cubicle)  # Connect the line to the 'from' busbar+
        shunt_gf.SetAttribute('e:ushnm', rated_kv)          # Connect the line to the 'to' busbar
        shunt_gf.SetAttribute('e:shtype', 2)      # shunt type = capacitor
        shunt_gf.SetAttribute('e:qcapn', dq_step)   # shunt value = dq_step (Mvar)
        shunt_gf.SetAttribute("outserv", 0)               # in service
        shunt_gf.SetAttribute('e:iswitch', True)
        shunt_gf.SetAttribute('e:qini', 1)  
        
        print({shunt})
 

        #shunt.SetAttribute('qini', dq_step)       # +Q injection (Mvar)
        '''

        load_name = gname + '_ld'
        load = pf_data.project.CreateObject('ElmLod', load_name)
        pf_data.grid_folder.Move(load)
        Node_contents = pf_data.grid_folder.GetContents(genbus)

        Node = Node_contents[0]

        Node_Cubicle = Node.CreateObject('StaCubic', gname + '_Ld_Cub')

        load_gf = pf_data.grid_folder.GetContents(load_name + '.ElmLod')[0]

        load_gf.SetAttribute('e:bus1', Node_Cubicle)  # Connect the line to the 'from' busbar
        load_gf.SetAttribute('plini', 0)
        load_gf.SetAttribute('qlini', 500)
        load_gf.SetAttribute('typ_id', None)  # allow direct P/Q entries

        gen = pf_data.grid_folder.GetContents(f"{gname}.ElmSym")[0]  # or ElmGenstat


        # ── switch to Const.Q with present schedule ─────────────────────
        gen.SetAttribute("av_mode", 'constq')     # 0=const V, 1=const Q  (check DB name)

        try:
            if ldf.Execute() != 0:
                print(" ! LF failed for", gname, "– skipping")
                continue

            # ---- ΔV metric ----
            if metric == "max":
                dv = max(abs(b.GetAttribute('m:u') - v_base[b.loc_name])
                         for b in buses)
            else:
                dv2 = sum((b.GetAttribute('m:u') - v_base[b.loc_name])**2
                          for b in buses)
                dv = dv2**0.5

            # ---- head-room weight (JSON) ----
            if weight_headroom:
                qmax = gdict.get("MVar_Max")
                qmin = gdict.get("MVar_Min")
                if (qmax is not None) and (qmin is not None):
                    q_sched = gdict.get("MVar_Sched", 0.0)
                    head = max(qmax - q_sched, q_sched - qmin)
                    if head == 0:
                        head = 1.0
                    dv *= head

            scores[gname] = dv
            print(f"   {gname:20s}  score = {dv:10.4e}")
            
            '''
            shunt_gf.SetAttribute('e:qcapn', 0)   # shunt value = dq_step (Mvar)
            shunt_gf.SetAttribute('e:qini', 0)  
            '''
            load_gf.SetAttribute('e:qlini', 0) 
            gen.SetAttribute("av_mode", 'constv')   # restore AVR

        except Exception: 
            print(f" ! Error during LF for {gname} – skipping")

    if not scores:
        raise RuntimeError("No scores computed – all LF runs failed.")

    # ------------------------------------------------------------------ #
    # 4) update JSON ................................................... #
    for gdict in snapshot["generators"]:
        gdict["VQ_score"] = scores.get(gdict["name"])

    ranked   = sorted(scores, key=scores.get, reverse=True)
    n_keep   = max(1, round(top_fraction * len(ranked)))
    top_set  = set(ranked[:n_keep])

    for gdict in snapshot["generators"]:
        gdict["selected_for_tuning"] = gdict["name"] in top_set

    snapshot.update({
        "VQ_analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        "VQ_method"           : "grid-shunt_injection",
        "VQ_dq_step_MVar"     : dq_step,
        "VQ_metric"           : metric,
        "VQ_weighted"         : weight_headroom,
        "VQ_top_fraction"     : top_fraction
    })

    with open(snap_path, "w", encoding="utf-8") as fp:
        json.dump(snapshot, fp, indent=2)

    print("Snapshot updated with V-Q scores →", snap_path)
