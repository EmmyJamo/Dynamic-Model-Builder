import json
from pathlib import Path

# adjust these to point at your specs
SPEC_FOLDER = Path(r"C:\Users\JamesThornton\source\repos\Dynamic Model Builder\Dynamic Model Builder\JSON_DB\Controllers")
SNAPSHOT_FOLDER = Path(r"C:\Users\JamesThornton\source\repos\Dynamic Model Builder\Dynamic Model Builder\JSON_DB\Network_Snapshots")
#PF_Snapshot.snapshot_network(PF_Data)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_spec(spec_id: str):
    spec_path = SPEC_FOLDER / f"{spec_id}.json"
    return load_json(spec_path)

def create_controller_from_spec(pf_data, spec: dict, gen_obj, gains: dict = None):
    """
    Instantiates the blocks defined in `spec` and wires them up to `gen_obj`.
    Identical to the earlier builder, but accepting a spec dict directly.
    """
    # default gains if none provided
    tunable = spec.get("tunable", {})
    if gains is None:
        gains = {k: v["default"] for k, v in tunable.items()}

    study = pf_data.app.GetActiveStudyCase()
    created = {}

    # 1) Create each block
    for blk in spec["blocks"]:
        blk_obj = study.CreateObject(blk["type"], blk["name"])
        for p, v in (blk.get("params") or {}).items():
            val = gains[v[2:-1]] if isinstance(v, str) and v.startswith("${") else v
            blk_obj.SetAttribute(p, val)
        created[blk["name"]] = blk_obj

    # 2) Wire connections
    for src, dst in spec["connections"]:
        src_name, src_attr = src.split(".")
        dst_name, dst_attr = dst.split(".")
        created[dst_name].Connect(dst_attr, created[src_name], src_attr)

    # 3) Attach last block to generator
    final_block = list(created.values())[-1]
    # assume generators use 'u:ctl' for their control attribute
    gen_obj.SetAttribute("u:ctl", final_block)

    return created

def build_controllers_for_snapshot(pf_data):
    """
    Reads <snapshot_name>_network_snapshot.json,
    finds each generator in the PF case by name,
    and attaches the appropriate controller.
    """
    # 1) Load the snapshot
    snap_path = SNAPSHOT_FOLDER / f"{PF_Data.project_name}_network_snapshot.json"
    data = load_json(snap_path)

    # 2) Define mapping from gen type ? controller spec ID
    spec_map = {
        "synchronous": "SG_AVR_V1",
        "inverter":    "Renewable_GF_V1"
    }

    # 3) Loop through each generator entry
    for gen_info in data["generators"]:
        name = gen_info["name"]
        kind = gen_info["type"]
        spec_id = spec_map.get(kind)
        if spec_id is None:
            print(f"  • Skipping {name}: no spec for type '{kind}'")
            continue

        # find the live PF object
        objs = pf_data.app.GetCalcRelevantObjects(f"*.{ 'ElmSym' if kind=='synchronous' else 'ElmGenstat'}")
        # match by loc_name
        gen_objs = [o for o in objs if o.loc_name == name]
        if not gen_objs:
            print(f"  • Could not find generator object named '{name}' in PF case")
            continue
        gen_obj = gen_objs[0]

        # load spec, instantiate controller
        spec = load_spec(spec_id)
        create_controller_from_spec(pf_data, spec, gen_obj)
        print(f"  ? Attached {spec_id} to {name} ({kind})")

    print("All done.")


build_controllers_for_snapshot(PF_Data)
