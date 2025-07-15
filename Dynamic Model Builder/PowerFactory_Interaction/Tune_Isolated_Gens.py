


def _tune_avr_parameters(pf_data, gen_meta: dict, iteration: int,
                         last_score: float) -> bool:
    """
    Modify AVR (or PSS, governor …) parameters of *this* generator.

    Return
    ------
    bool
        True  → parameters were changed, try another simulation run.
        False → nothing changed (or you give up) → loop stops.
    """
    # ⚠️  REPLACE THIS WITH YOUR REAL TUNING LOGIC  ⚠️
    # ------------------------------------------------------------------
    print(f"   ↪ (placeholder) no AVR changes applied, score = {last_score:.5f}")
    return False                # stop after the very first iteration
