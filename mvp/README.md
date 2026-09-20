# MVP: SAM-Road post-processing lab (GPU-free)

## Run
```bash
pip install numpy scipy scikit-image opencv-python
python roadgap_lab.py --n_val 8 --n_test 24      # ~10-15 min on a laptop CPU (bridge search is the slow part)
```
Output is printed and saved to `outputs/results_synthetic.txt`.

## What it is / is not
- **Is:** a controlled sandbox that checks the *mechanisms* behind the research gap:
  1. gamma on a probability mask == a threshold shift (asserted numerically),
  2. a P-only "adaptive gamma" is also just a threshold shift,
  3. distance-only gap bridging (Paper 1 style) vs direction-aware + evidence-verified bridging (ours),
  4. a validation-tuned / test-reported protocol with bootstrap confidence intervals.
- **Is not:** evidence of real TOPO/APLS gains. Metrics here are **proxies** (`APLS-lite`, buffer precision/recall/F1, `GapRec`). Real numbers need the official TOPO/APLS code and real SAM-Road masks.

## Families
| Tag | Meaning |
|---|---|
| A | baseline: `P > 0.5` |
| B | Paper 1: global gamma on mask, then threshold |
| C | control Paper 1 never ran: plain threshold sweep |
| D | other AI's Gap 1: `gamma = 1 + a(1-P)` (P-only) |
| E | Paper 1 A* emulation: half-plane rule, `MIN_GRAPH = 2*MAX`, no verification |
| F | Paper 1 hybrid: B + E |
| G0/G1/G2 | ours: evidence-only / direction-only / direction + evidence (2x2 ablation) |

Bridge step cost: `len * (1 + w_ev*(1-P)) * (1 + lam*(1-cos(turn)))`, search state = (y, x, heading), no U-turns when `lam > 0`. A bridge is accepted only if mean mask probability along it is `>= p_min`. `lam` and `p_min` are chosen on the validation worlds.

## Plugging in real SAM-Road (Week 1-3 of the plan)
SAM-Road's `inferencer.py` builds `fused_road_mask` / `fused_keypoint_mask` (float, later rescaled to 0..255) and then calls `graph_extraction.extract_graph_points(...)` **[confirm names in your checkout]**.
1. Save the fused masks per test image (`np.save`).
2. **Controls:** re-run graph extraction with `mask ** (1/gamma)` and with lowered `ROAD_THRESHOLD` / `ITSC_THRESHOLD`; compare with the official TOPO/APLS scripts.
3. **Bridging:** take the extracted graph's degree-1 nodes as dead ends (instead of skeleton end-pixels), run `find_bridges` on the road mask (`Pm = mask/255`), keep bridges that pass `p_min`, add them as edges, re-score.
4. Pick `lam`, `p_min`, `R` on the **validation** split; freeze; run once on test.
