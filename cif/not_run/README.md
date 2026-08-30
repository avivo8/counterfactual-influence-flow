# Implemented but never executed

These scripts are complete and were part of the design, but **produced no result reported
anywhere in this repository**. They are kept because the plans reference them and because
silently deleting unrun work would misrepresent what was done — not because they are validated.

| file | status |
|---|---|
| `validate_if.py` | The one-step-influence vs real-retraining check (Milestone 2). Written, memory-corrected, **never run** — the machine could not afford it alongside the campaign. The single largest open item: it is what would validate the influence approximation itself. |
| `repr_analysis.py` | Misalignment-direction / layerwise-representation analysis. Superseded by `geometry.py`, which answers the same question with a stronger control design. |
| `unrelated_control.py` | Standalone unrelated-concept control. Superseded by family **F3** inside `structured_controls.py`, which runs the same contrasts as part of the full 124-control ensemble. |

Deleted rather than kept, for the record: `analysis.py` (1,775 lines, dead — nothing imported
it and no reported number came from it), `tune.py` (swept `mode="ihvp"` with λ ∈ {1e-3,1e-2,1e-1},
a regime the curvature measurements showed cannot work), and `eval_all.py` (superseded by
`eval_fast.py`, which fixed the probe-subsampling bug that would have corrupted R_B).
