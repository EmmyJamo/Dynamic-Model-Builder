def _build_single_ib(meta, pf_data):
    """
    Build a single-machine-infinite-bus (SMIB) around one generator.
    Adds verbose printouts and full try/except blocks for easier debugging.
    """
    print(f"\n=== SMIB build for «{meta['name']}» started =======================")

    try:
        # -------- gather input data --------------------------------------
        gname   = meta["name"]
        kv      = meta.get("voltage")
        phi_deg = float(meta.get("phiui", 0.0))
        RtoX    = meta.get("RtoX")
        Skss    = meta.get("Skss_MVA")
        Ikss    = meta.get("Ikss_kA")
        typ     = meta.get("type")
        pow_fac = float (math.cos(math.radians(phi_deg)))

        project = pf_data.project  # NOTE: use the PF project object
        grid    = pf_data.grid_folder

        print(f"• inputs  : kv={kv}, phi={phi_deg}°, Skss={Skss} MVA, Ikss={Ikss} kA")
        print(f"• gen type: {typ},  R/X = {RtoX},  cosφ = {pow_fac:.4f}")

        # -------- create terminal & infinite bus -------------------------
        print("→ creating ElmTerm and ElmXnet …")
        infbus_bb = project.CreateObject('ElmTerm',  'infbus_BB')
        Genbus = project.CreateObject('ElmTerm',  gname + '_BB')
        ibus = project.CreateObject('ElmXnet',  gname + '_IB')
        CB   = project.CreateObject('ElmCoup', gname + '_CB')

        if not infbus_bb or not ibus or not Genbus:
            raise RuntimeError("failed to create ElmTerm or ElmXnet")

        grid.Move(infbus_bb); grid.Move(ibus) ; grid.Move(Genbus) ; grid.Move(CB)  # move to grid folder

        # -------- terminal & cubicles -----------------------------------
        infbus_bb.SetAttribute('e:uknom', kv)
        Genbus.SetAttribute('e:uknom', kv)  

        infbus_bb_cb1 = infbus_bb.CreateObject('StaCubic', gname + '_IB_C1')
        infbus_bb_cb2 = infbus_bb.CreateObject('StaCubic', gname + '_IB_C2')

        Genbus_cb1 = Genbus.CreateObject('StaCubic', gname + '_BB_C1')
        Genbus_cb2 = Genbus.CreateObject('StaCubic', gname + '_BB_C2')


        if not infbus_bb_cb1 or not infbus_bb_cb2 or not Genbus_cb1 or not Genbus_cb2:
            raise RuntimeError("failed to create StaCubic objects")

        # -------- wire the infinite bus ---------------------------------
        print("→ wiring infinite bus attributes …")
        ibus.SetAttribute('e:bus1',  infbus_bb_cb1)
        ibus.SetAttribute('e:bustp', "SL")
        ibus.SetAttribute('e:cused', 0)
        ibus.SetAttribute('e:snss',  Skss)
        ibus.SetAttribute('e:ikss',  Ikss)
        ibus.SetAttribute('e:rntxn', RtoX)
        ibus.SetAttribute('e:cosn',  pow_fac)

        # -------- wire the Circuit Breaker cubicle ---------------------------
        print("→ wiring Circuit Breaker cubicle …")
        CB.SetAttribute('e:bus1', infbus_bb_cb2)  # connect to infinite bus cubicle
        CB.SetAttribute('e:bus2', Genbus_cb2)     # connect to generator cubicle
        CB.SetAttribute('e:on_off', True)            # set voltage level

        # -------- point the generator to cubicle #2 ----------------------
        print("→ attaching generator to cubicle #2 …")
        if typ == "synchronous":
            gen = grid.GetContents(gname + ".ElmSym")[0]
        else:
            gen = grid.GetContents(gname + ".ElmGenstat")[0]

        gen.SetAttribute('e:bus1', Genbus_cb1)
        gen.SetAttribute('e:av_mode', 'constq')
        gen.SetAttribute('e:mode_inp', 'PQ')  

        print(f"✅  SMIB build for «{gname}» completed successfully")

    except Exception as exc:
        print(f"⛔  ERROR while building SMIB for «{meta.get('name','?')}» : {exc}")
        traceback.print_exc()



# 3) build SMIB inside active variant
try:
    _build_single_ib(meta, pf_data)
    print(f"✅  {gname}: SMIB ready in «{var_id}»")
except Exception:
    print(f"⚠️  {gname}: failed")
    traceback.print_exc()
