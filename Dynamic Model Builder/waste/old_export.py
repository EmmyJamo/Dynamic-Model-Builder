# ------------------------------------------------------------------
# inside class RMSRunner            (replace the old _export method)
# ------------------------------------------------------------------
def _export(self, study_case, elmres, exp: ExportOptions):
    """
    Export the active result file in the requested formats.
    • CSV is always tried           (code 6)
    • XLSX is tried only if the code is accepted by this PF session
    """
    fmt_codes = {".csv": 6, ".xlsx": 8}          # <─ adjust if needed

    def try_export(ext: str, target: Path):
        if not target:
            return                              # nothing requested
        code = fmt_codes[ext]

        comres = study_case.CreateObject(
            "ComRes", f"Export_{ext}_{datetime.datetime.now():%H%M%S}"
        )
        # set attributes via the generic interface (= less picky)
        ok  = (
            comres.SetAttribute("pResult", elmres) == 0 and
            comres.SetAttribute("f_name",  str(target)) == 0 and
            comres.SetAttribute("iopt_exp", code) == 0 and
            comres.SetAttribute("iopt_locn", 1) == 0 and   # English decimal
            comres.SetAttribute("ciopt_head", 1) == 0      # header row
        )
        if not ok or comres.Execute() != 0:
            # most likely the export format is not licensed – warn & skip
            print(f"⚠️  Cannot export {ext} (code {code}) – skipped.")
        comres.Delete()                     # tidy up

    # ------------------------------------------------------------------
    try_export(".csv",  Path(exp.csv_path).with_suffix(".csv") if exp.csv_path else None)
    try_export(".xlsx", Path(exp.xlsx_path).with_suffix(".xlsx") if exp.xlsx_path else None)


