# Road-graph extraction from satellite imagery — deep-dive report, verified research gap, and MVP

*Prepared 2026-09-21. Everything below was checked against your five PDFs, the SAM-Road / SAM-Road++ / DOGE sources, and the SAM-Road code on GitHub. Items I could not verify are marked **[UNVERIFIED]**.*

---

## 0. Read this first (the 60-second version)

**Main paper = Paper 1** (Fathul'ibad et al., *JISEBI* 12(1), Feb 2026). It is a good, doable base — but **the other AI's write-up contains several errors and one fatal flaw in its "Gap 1"**. Verified corrections:

| # | Other AI claimed | What the evidence says |
|---|---|---|
| 1 | "Gap 1: entropy-guided local gamma `γ(x,y)=1+α(1−P)` is *spatially adaptive*" | It is **not**. `γ` depends only on `P`, so `P' = P^(1/γ(P))` is just another **pointwise, monotone tone curve**. Any pointwise monotone remap followed by a threshold is *exactly* equivalent to a **different threshold**. I checked it numerically (mismatch = 0.000000, see §8). As written it cannot beat "just lower the threshold". |
| 2 | "Global gamma over-saturates clear road corridors" | On a **probability mask**, `x^(1/γ)` keeps 0→0 and 1→1. Confident pixels barely move. The "bloat" comes from the **low-confidence fringe crossing the threshold** (mask dilation), not from saturating clear roads. |
| 3 | "Gap 3 target: outperform DOGE's City-scale APLS (70.24)" | **Paper 1 already does** (71.19 vs 70.24, its Table 6). The real hard targets are DOGE **SpaceNet APLS 73.48** and **SAM-Road++ SpaceNet APLS 73.44** (numbers from the DOGE arXiv tables). |
| 4 | "Differentiable soft-skeleton before A*" | Differentiability is only useful for *training*. At inference a plain `skimage.skeletonize` does the same job. Also, per the SAM-Road repo, its pipeline extracts nodes by NMS on the masks and (in `graph_extraction.py`) runs A* on a `255 − road_mask` cost field — no skeleton step **[verify in the inference path]**. |
| 5 | "Baseline C: ISPRS IJGI 2026 direction-aware Dijkstra + PSPNet" | I could **not locate it** in any search. **[UNVERIFIED — do not cite until you have a DOI.]** |
| 6 | "Table 4 / −2.42 APLS at the +1.39 TOPO setting" | −2.42 APLS is SpaceNet γ=2.2 (Table 2). The +1.39 TOPO setting (City-scale γ=2.0) comes with **APLS −1.86**. |
| 7 | "2026 + zero citations = untouched, first-mover" | Mostly true for Paper 1 (Crossref citations = 0). **But** an MDPI *Remote Sensing* paper published **15 Sep 2026** — *Occlusion-Aware Topology Refinement for Robust Road Graph Extraction from Satellite Imagery* — attacks the same SAM-Road occlusion problem (training-based: synthetic occlusion augmentation + occlusion-adaptive extended-line + hard-mining). The area is live. |

**Our recommended gap (details §7):** *a training-free, path-evidence-verified (and, as an ablation, direction-aware) gap-bridging stage for SAM-Road, tuned on a validation split, judged on whether ONE configuration improves TOPO and APLS together, and evaluated on occlusion-stratified segments* — plus the missing control experiment "gamma vs plain threshold sweep". In the synthetic MVP the evidence verifier carried the gain; the direction prior did not add measurable value yet.

**MVP built:** [`mvp/roadgap_lab.py`](mvp/roadgap_lab.py) (GPU-free, runs on your laptop). Results in §8.

---

## 1. Definitions — with real-world examples, where they appear in Paper 1, and the technical side

### 1.1 Satellite imagery & resolution
- **What:** Top-down photos of Earth. *Spatial resolution 1 m/pixel* = one pixel covers 1 m × 1 m.
- **Real world:** Google Maps satellite view. At 1 m/px a car is ~4 pixels; a road is ~5–10 pixels wide.
- **In Paper 1:** §II.A. SpaceNet tiles are 400×400 px (0.4 km × 0.4 km); City-scale tiles are 2048×2048 px (2 km × 2 km), both at 1 m/px. This is why the algorithm parameters "8 / 16 / 32 **m**" are also "8 / 16 / 32 **pixels**".
- **Technical:** Images are RGB arrays `[H, W, 3]`. Big tiles are processed with a sliding window.

### 1.2 Road network graph (nodes and edges)
- **What:** A road map as a mathematical graph. **Node** = intersection or bend point; **edge** = road piece between two nodes.
- **Real world:** A subway map — dots (stations) and lines (tracks). A GPS app routes by walking this graph.
- **In Paper 1:** Introduction & Fig. 5. SAM-Road's *output* is this graph, not a picture.
- **Technical:** `networkx`-style structure (or a pickled dict of vertices/edges in SAM-Road's `graph/` output folder).

### 1.3 Semantic segmentation vs. graph extraction
- **Segmentation:** label every pixel "road / not road" (a mask). **Graph extraction:** output the *connected map*.
- **Real world:** Colouring a map (segmentation) vs. drawing the metro lines with stations (graph).
- **In your papers:** Paper 2 (U-Net, IJECES) = segmentation only. SAM-Road, kLCRNet, DOGE = graph output.
- **Why it matters:** a 99 %-pixel-accurate mask with one 3-px hole at a junction gives a graph a car **cannot drive through**.

### 1.4 Occlusion
- **What:** Something blocks the view of the road: tree canopy, building shadow, overpass, cloud.
- **Real world:** A tree branch over a lane in a photo — the road "disappears" for 20 m and re-appears.
- **In Paper 1:** the *entire motivation* (Abstract, §I). Note: Paper 1 never **measures** occlusion (no occlusion labels or occlusion-stratified scores — see §6).
- **Technical:** appears as low road probability (~0.1–0.4) in the middle of a road, so thresholding cuts the road in two.

### 1.5 Foundation model, SAM, ViT
- **Foundation model:** a huge model pre-trained on massive data that you adapt to many tasks. **SAM** (Segment Anything Model, Meta) is one for image segmentation. Its image encoder is a **ViT** (Vision Transformer): it chops an image into 16×16 patches and lets every patch "look at" every other patch (self-attention).
- **Real world:** Someone who has seen a billion photos; you only teach them "now find roads".
- **In Paper 1:** SAM-Road = SAM adapted to roads (ref. [18]). Paper 1 does **not** train or change it.
- **Technical:** SAM-Road uses the smallest ViT-B (~80 M parameters) and **fully fine-tunes** the encoder at 0.1× learning rate (per the SAM-Road paper).

### 1.6 SAM-Road (the model being improved) — the "how it works" box
Per the SAM-Road paper (arXiv 2403.16051, CVPRW 2024 best paper) and repo `htcr/sam_road`:
1. **Image encoder** (ViT-B) → feature map at 1/16 resolution.
2. **Geometry decoder** — 4 transposed-conv layers → two probability maps at full resolution: **road mask** and **intersection/keypoint mask** (values 0–1, stored as 0–255).
3. **Vertices** — pixels above `ROAD_THRESHOLD` / `ITSC_THRESHOLD` → **non-maximum suppression** (`nms_points`, radius ≈16 px) → graph nodes. Intersections get priority.
4. **Topology decoder** — a small transformer (3 self-attention layers; described as a GNN) that takes image embeddings at two candidate nodes + their relative offset and outputs the probability that a road edge connects them. Candidate pairs = nearest neighbours within `NEIGHBOR_RADIUS`; kept if `> TOPO_THRESHOLD`.
5. Large images: **sliding-window** inference with overlaps; masks fused by averaging.
- **Trained by:** the *original* authors (single RTX 4090, Adam, BCE loss, 90° rotations). Paper 1 downloads their **pre-trained checkpoints** (Hugging Face `congrui/sam_road`).

### 1.7 Probability mask, sigmoid, threshold
- **What:** The model outputs a confidence per pixel; a **threshold** (say 0.5) turns confidence into yes/no.
- **Real world:** Airport metal-detector sensitivity: set high → misses a small item; set low → false alarms on belt buckles.
- **In Paper 1:** the **gamma correction is applied to this mask** (Fig. 4: divide by 255 → raise to 1/γ → multiply by 255).
- **Technical (important!):** because thresholding follows, `x^(1/γ) > τ  ⇔  x > τ^γ`. So for the threshold step **gamma correction = lowering the threshold** (τ=0.5, γ=2 → τ'=0.25). Paper 1 never compares against a plain threshold sweep. **This is the single biggest missing control in the paper.**

### 1.8 Gamma correction
- **What:** A non-linear brightness curve `out = in^(1/γ)`. γ>1 (in this inverse form) brightens dark values much more than bright ones.
- **Real world:** The "brightness/shadows" slider on a phone photo editor — lifts dark areas, leaves whites alone.
- **In Paper 1:** §II.B, Eq. (1), Fig. 3–4, Table 2. Tested γ ∈ {1.25, 1.5, 1.75, 2.0, 2.2}, applied globally (same γ everywhere).
- **Technical:** one NumPy line `mask = (mask/255) ** (1/gamma) * 255`.

### 1.9 Inference vs. training; pre-trained weights
- **Training** = teaching the network (needs labelled data + big GPU). **Inference** = using the trained network on new images. **Pre-trained weights** = someone else's finished lessons.
- **In Paper 1:** the entire study is **inference-only** on a free Colab 16 GB GPU (§II.D). *Nothing was trained.* This is why the work is reproducible on student hardware.

### 1.10 A* and Dijkstra (shortest-path search)
- **What:** Algorithms that find the cheapest route through a grid/graph. **Dijkstra** explores outward evenly; **A\*** adds a *heuristic* (an optimistic guess of the remaining distance) to aim at the goal.
- **Real world:** Google Maps choosing the fastest route; the heuristic is "straight-line distance to destination".
- **In Paper 1:** §II.C + Algorithm 1 — **A\*** is used to **reconnect road ends** ("dead ends") that occlusion broke. Adapted from DeepRoadMapper (Mattyus et al., ICCV 2017).
- **How exactly (Algorithm 1):** start from a dead-end pixel; search only inside a window of radius `MAX_STRAIGHT_DISTANCE`; only expand pixels *closer to the start than to a reference point `opp` behind the dead end* (a **half-plane "keep going forward" rule**); cost = movement cost + segmentation penalty; stop when the path reaches a road pixel that is at least `MIN_GRAPH_DISTANCE` away *along the graph*. The paper uses `MIN = 2×MAX` → the tested pairs **(16,8), (32,16), (64,32) m** are one tied 3-point grid, not two independent knobs.
- **Note:** SAM-Road's own `graph_extraction.py` already contains an A*-style routine (`create_cost_field_astar`, `is_connected_astar`, cost `255 − road_mask`) **[verify which routine the released inference path calls]**. Paper 1's "modified A*" is an **extra** dead-end reconnection pass.

### 1.11 TOPO metric ("marbles and holes")
- **What:** Introduced by Biagioni & Eriksson (2012). Drop "marbles" along the *predicted* graph and check if each lies near the ground-truth (matched vs **spurious**); drop "holes" along the *ground truth* and check if the prediction covers them (matched vs **missing**). Precision = matched/(matched+spurious), Recall = matched/(matched+missing), **TOPO F1** = their harmonic mean.
- **Real world:** Spray-paint dots on your route and check they are on a real road; then check that every real road got a dot.
- **In Paper 1:** Eq. (2)–(4), all tables. TOPO **precision drops when A* invents fake roads** ("spurious marbles").
- **Technical:** official implementation lives in the Sat2Graph / SAM-Road repos (Python).

### 1.12 APLS (Average Path Length Similarity)
- **What:** Pick many pairs of points; compare the *shortest path length* between them in ground truth vs. prediction. `APLS = 1 − mean(min(1, |L_gt − L_pred| / L_gt))`. Missing connection → path doesn't exist → score 0 for that pair. **Fake shortcuts** also lower it (path too short).
- **Real world:** "Does my app's driving distance A→B match the real one?"
- **In Paper 1:** Eq. (5); reported as the "geometric/routing" metric, TOPO as the "connectivity" metric.
- **Technical:** the official implementation is in **Go** (the SAM-Road repo needs Go installed to compute APLS).

### 1.13 Trade-off / Pareto improvement
- **Trade-off:** improving A worsens B. **Pareto improvement:** improves at least one metric and worsens none.
- **In Paper 1:** the Discussion admits a "fundamental trade-off" TOPO ↔ APLS. **Key point for us:** Table 5's "+0.31 TOPO, +0.49 APLS" (SpaceNet) and "+1.39, +3.58" (City-scale) each mix numbers from **different configurations** (see §6). The hybrid never shows a large single-configuration win on both.

### 1.14 Concepts from the other three papers (for your own reading)
| Term | Plain meaning | Where |
|---|---|---|
| **U-Net** (encoder-decoder + skip connections) | Shrink the image to understand it, expand back to draw the mask, with "shortcut wires" that keep fine detail | Paper 2 (IJECES) |
| **clDice / soft-clDice** | A training loss that compares *skeletons* (centrelines) so a broken thin road is punished heavily | SAM2-RoadNet, §3.5 |
| **Adapter** | Tiny trainable layers inserted into a frozen big model — cheap domain adaptation | SAM2-RoadNet |
| **RFB / W-BiFPN** | Multi-scale feature blocks: look at a road with several "zoom levels" and learn how to mix them | SAM2-RoadNet |
| **Keypoint detection + bipartite (Hungarian) matching** | Detect road "dots" as heat-spots, then match predicted dots to true dots one-to-one | kLCRNet |
| **Local connectivity exploration** | For each dot, classify which nearby dots it links to | kLCRNet |

---

## 2. Architecture of the research (Paper 1), end to end

```
           ┌─────────── done once by the ORIGINAL SAM-Road authors (not Paper 1) ───────────┐
           │ SpaceNet / City-scale TRAIN split ──► train ViT-B encoder + geometry decoder   │
           │                                        + topology decoder  ──► checkpoint      │
           └────────────────────────────────────────────────────────────────────────────────┘
                                                   │  (pre-trained weights, HF hub)
                                                   ▼
Paper 1 (inference-only, Colab 16 GB GPU, TEST split only: 382 SpaceNet / 27 City-scale images)
 ┌───────────┐   ┌────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │ satellite │──►│ SAM-Road           │──►│ ① GAMMA          │──►│ SAM-Road graph       │
 │ image     │   │ road + intersection│   │ x^(1/γ) on the   │   │ extraction: threshold│
 │           │   │ probability masks  │   │ predicted mask   │   │ → NMS nodes → edges  │
 └───────────┘   └────────────────────┘   │ (γ=1.25…2.2)     │   └──────────┬───────────┘
                                          └──────────────────┘              ▼
                                                              ┌──────────────────────────┐
                                                              │ ② MODIFIED A*            │
                                                              │ reconnect dead ends,     │
                                                              │ (MIN/MAX = 16/8, 32/16,  │
                                                              │  64/32 m)                │
                                                              └────────────┬─────────────┘
                                                                           ▼
                                                    TOPO F1 + APLS vs ground truth
                                                    (Tables 2–6; compare with RNGDet++, DOGE)
```
**How the definitions are used:** occlusion (motivation) → SAM-Road masks (§1.6) → gamma on masks (§1.8, ≡ threshold shift §1.7) → A* dead-end reconnection (§1.10) → TOPO/APLS (§1.11–1.12) → discussion of the trade-off (§1.13).

---

## 3. The problem, and what Paper 1 actually solved

### Baby-simple version
A computer looks at a satellite photo and draws the roads. Where trees or shadows cover a road, the computer's drawing has a **hole**, so a navigation app can't drive through. The authors did **two cheap tricks and no re-training**: (1) turn up the computer's "confidence" in dim parts of the drawing so faint road pieces count as road, and (2) draw a bridge wherever a road stops suddenly and there's another road nearby. Result: a bit fewer holes — but sometimes the bridges go to the wrong place.

### Jargon version
SAM-Road's threshold-and-NMS vertex extraction leaves **fragmented components** under occlusion, lowering TOPO recall and APLS reachability. Paper 1 proposes **training-free inference-time post-processing**: (i) **global inverse-gamma** tone-mapping of the predicted probability mask to raise the low-confidence tail above the extraction threshold; (ii) a **half-plane-constrained A\*** dead-end reconnection with `MIN_GRAPH_DISTANCE`/`MAX_STRAIGHT_DISTANCE` gating and a segmentation-probability penalty. It evaluates a 5-point γ sweep, a 3-point A* sweep and their product on the SpaceNet and City-scale **test** splits with TOPO-F1 and APLS.

### What it actually establishes (verified from Tables 2–5)
- Training-free post-processing **can** move the metrics on top of a frozen SAM-Road, and the direction of the trade-off is documented:
  - γ↑ ⇒ TOPO↑ (up to +0.33 SpaceNet, +1.39 City-scale) but APLS↓ (as low as −2.42 / −1.96).
  - A* with long range: City-scale APLS **+3.33** at (64,32) but SpaceNet TOPO **−3.52** (Table 3).
- Best-cell claims (Table 5): SpaceNet TOPO 80.59 / APLS 71.17; City-scale TOPO 78.56 / APLS 71.19.

### What it did **not** solve (this is where our gap is → §6, §7)
The paper leaves the TOPO↔APLS trade-off **open**, tunes on the test set, never runs the threshold control, never measures occlusion, and omits SAM-Road++.

---

## 4. Datasets, models, and what was trained

### Paper 1 (JISEBI 2026) — **trained nothing**
| Item | SpaceNet | City-scale |
|---|---|---|
| Source | DigitalGlobe WorldView-3; Las Vegas, Paris, Shanghai, Khartoum | Google Static Maps imagery; OpenStreetMap ground truth; 20 US cities, 720 km² |
| Size (full) | 2,549 tiles of 400×400 px | 180 tiles of 2048×2048 px |
| **Used by Paper 1** | **382 test tiles** | **27 test tiles** (SAM-Road paper says 29) |
| Resolution | 1 m/px | 1 m/px |
| Model | *Frozen* pre-trained SAM-Road ViT-B (trained by original authors: 512² patches / batch 16 for City-scale, 256² / 64 for SpaceNet; Adam; BCE; single RTX 4090) | same |
| Baseline re-run in Colab | TOPO 80.28, APLS 70.68 | TOPO 77.17, APLS 67.61 |

⚠ The SAM-Road arXiv v3 (as I retrieved it) lists SpaceNet **80.52 / 71.64** and City-scale **77.23 / 68.37**. Paper 1's re-run baselines are slightly lower, so on SpaceNet its "best" APLS **71.17 is still below SAM-Road's own published 71.64**. **[Check which arXiv version Paper 1 used.]**

### The other four PDFs in your folder
| Paper | Venue / status | Task & model | Data & training | Note for us |
|---|---|---|---|---|
| **Paper 2** U-Net (Jabbar & Lateef) | IJECES 17(5), 2026 (journal) | plain U-Net, pixel masks | DeepGlobe-based, 13,004 train / 1,234 val / 1,101 test images; Adam 1e-3, batch 16, **10 epochs**, BCE+Dice | Low rigour: IoU is 66.62 in the abstract but 64.62/66.61 elsewhere; "precision 71.11" in a table row is the F1; text says DeepGlobe + AIcrowd + self-annotated but abstract says DeepGlobe. **No graph/topology** → not our base. |
| **SAM2-RoadNet** (Feng et al.) | *Remote Sensing* 18(6):913, Mar 2026 (journal); 0 citations | SAM2 encoder + adapters + RFB + W-BiFPN + **soft-clDice** loss | DeepGlobe, Massachusetts; AdamW 1e-4, batch 16, **50 epochs on 2×RTX 4090** | Needs **training** ⇒ heavier. Pixel-mask metrics (F1/IoU), not TOPO/APLS. |
| **kLCRNet** (Zhang et al.) | IEEE JSTARS 18, 2025 (journal) | HRNet-W32 keypoint heat-map + Hungarian matching + local-connectivity classifier | City-scale + SpaceNet; TOPO-F1 78.27 / APLS 68.96 (City-scale); 25× faster than TERNformer, 102× than RNGDet | Needs training. Its stated limits are intricate scenes; the "fails on canopy occlusion" claim in the other AI's text is **not in the paper**. |
| **Review** (Wang et al.) | *J. Traffic & Transp. Eng.* 2016 | classical review | – | **Not recent (2016).** Fine for background only. |

---

## 5. Answer to "does a high citation count make a gap easier or harder?"
Neither directly. What actually predicts "easy to improve": (a) **runs on student hardware**, (b) **code + checkpoints exist**, (c) the paper documents a **measurable weakness**, (d) **evaluation is standardised** (TOPO/APLS). Paper 1 satisfies all four (SAM-Road: MIT license, HF checkpoints, official metrics). Citation facts I could check: Paper 1 → Crossref citations **0**; SAM2-RoadNet → OpenAlex **0**. Caveat: 0 citations at 6 months is normal lag; it is **not** proof nobody is working on it (see the 15 Sep 2026 *Remote Sensing* paper). JISEBI: Scopus-indexed, **SJR Q2 (2025 update)**, SINTA S1, DOAJ (per UNAIR's announcement) — fine for a "non-conference journal" requirement, but it's an information-systems venue, not a remote-sensing one, which is partly why Paper 1's protocol is loose (good for us, but a stricter venue will demand more).

---

## 6. Verified weaknesses in Paper 1 (each is a research handle)

1. **Headline numbers are column-wise maxima from different cells.** Table 4 SpaceNet: best-TOPO cell (γ=1.5, 16/8) has APLS **−0.89**; best-APLS cell (γ=1.25, 32/16) has TOPO **−0.79**. City-scale: TOPO-best (γ=2, 16/8) APLS **−0.64**; APLS-best (γ=1.75, 64/32) TOPO **−0.33**. Table 5 then reports "+0.31 / +0.49" and "+1.39 / +3.58" as if they were one method. The only *single* settings in the paper that improve both metrics are **γ=1.25 alone** (Table 2: +0.27/+0.23 SpaceNet, +0.84/+1.12 City-scale) and **A\* 32/16 alone on City-scale** (Table 3: +0.02/+2.84). Among the hybrid cells the paper prints, none beats γ=1.25-alone on both metrics — so **the hybrid's benefit at a single operating point is never demonstrated**, and the SpaceNet gains are tiny (+0.27/+0.23). ⇒ *Pareto* gap.
2. **Hyper-parameters chosen on the test set** (only test splits were used, §II.A). No validation split, no held-out confirmation.
3. **No statistics.** SpaceNet gains (+0.31 / +0.49) come from 382 images with no variance, confidence interval or per-image analysis.
4. **Gamma ≡ threshold shift, control missing.** Never compared with plain sweeps of `ROAD_THRESHOLD`, `ITSC_THRESHOLD` (or `TOPO_THRESHOLD`). Also unclear **which** mask (road/intersection/both, pre- or post-fusion) the gamma is applied to; the Introduction says "before segmentation" but §II.B applies it to the *predicted mask*.
5. **A\* parameters are tied and global.** `MIN = 2×MAX`, 3-point grid, and the best setting **flips by dataset** (SpaceNet prefers 16/8; City-scale prefers 64/32). No per-gap decision rule; no accept/reject verification → spurious "marbles" at long range (TOPO −3.52).
6. **Only a half-plane direction rule** (`closer to start than to opp`). No angular/curvature cost, no path-evidence check.
7. **Occlusion is never measured.** No occlusion-stratified metric; "handles occlusion" is inferred from overall TOPO/APLS.
8. **Missing baseline: SAM-Road++ (CVPR 2025)** — which Paper 1 cites as [21] — reports SpaceNet APLS **73.44**, City-scale TOPO **80.01** (DOGE's tables). SAM-Road++ checkpoints are **not released** (README TODO) ⇒ we stay on SAM-Road.
9. **Comparison Table 6 is selective:** the proposed method loses to DOGE on 3 of 4 cells, and loses to RNGDet++ on SpaceNet TOPO (82.51 vs 80.59).
10. **Presentation errors** (helpful for spotting weak review): "Figure 4/5" mis-referenced in the Discussion; Table 5/6 layout garbled; no code link (Data Availability lists only the dataset).

---

## 7. **Our research gap** (recommended) and why it is defensible

### Title-style statement
> *Training-free, direction-aware and evidence-verified dead-end bridging for SAM-Road road graphs: closing the TOPO–APLS trade-off with one validation-tuned configuration, evaluated on occlusion-stratified road segments.*

### What is new relative to Paper 1 and the literature
| Piece | Status of prior art | Our contribution |
|---|---|---|
| Direction/tangent priors for road-gap linking | Old idea (tensor voting, DeepRoadMapper, SAM-Road++ *extended-line*) | Not the novelty by itself; **MVP did not show a gain from it alone** — keep as an ablation |
| **Accept/reject verification of every bridge** using path evidence (mean/min mask probability along the path, length, turn angle) | Not in Paper 1 (accept-any-path). I found no training-free version on SAM-Road **[searched, not exhaustive]** | **Core novelty**: removes the spurious-shortcut failure (Table 3's −3.52) |
| **Pareto-style protocol**: choose ONE config on a *validation* split; report ΔTOPO & ΔAPLS with bootstrap CIs on test | Absent in Paper 1 | **Methodological novelty** |
| **Occlusion-stratified evaluation** ("gap recall / gap precision": recover-rate on GT road segments that lie under canopy/shadow) | Absent | New **metric/analysis** (software-only) |
| **Gamma-vs-threshold control** (and a *context-conditioned* gain instead of P-only gamma) | Absent | Either confirms or refutes Paper 1's gamma contribution — publishable either way |

### Hypotheses (testable)
- **H1** A plain threshold sweep reproduces most of Paper 1's gamma gains (gamma is not more than a threshold shift).
- **H2** Path-evidence verification (and, to be tested, a direction prior — *unproven in the synthetic MVP*) recovers long-range bridging benefit (City-scale APLS at 64/32) **without** the SpaceNet TOPO collapse.
- **H3** One validation-selected configuration improves **both** TOPO and APLS over frozen SAM-Road **and over γ=1.25-alone** (the only both-metric winner in Paper 1) with 95 % bootstrap CI excluding 0.
- **H4** Gains concentrate on occluded segments (gap recall ↑ while precision stays flat).

### Honest risks
- **Scoop / overlap:** *Occlusion-Aware Topology Refinement…* (Remote Sensing 18(18):3173, 15 Sep 2026) uses an occlusion-adaptive extended-line plus training-time tricks on SAM-Road++. We differ (training-free, verifier, protocol) but **must cite and differentiate**.
- **Incrementalness:** post-processing papers are easy to dismiss unless the evaluation is rigorous — that is why the protocol + stratified metric + control are part of the contribution.
- **Real-data uncertainty:** the MVP below is on a synthetic world; it proves mechanisms, **not** real TOPO/APLS gains.
- **Realistic targets:** do **not** promise beating DOGE (SpaceNet APLS 73.48) with post-processing on SAM-Road. Promise: *Pareto improvement over the SAM-Road baseline and over Paper 1's hybrid, with statistics.*

### What to drop from the other AI's list
- Gap 1 **as written** → replace with the control (H1) and, optionally, a *neighbourhood/endpoint-conditioned* gain (needs spatial context, e.g. distance/alignment to dead ends or a vegetation/shadow cue from RGB).
- Gap 3 (soft-skeleton) → optional low-priority ablation (`skimage.skeletonize`), not a headline.

---

## 8. MVP (≈20–25 % of the full execution) — what exists and what it shows

**Work-breakdown (my estimate):** baseline reproduction 10 % · controls (gamma/threshold) 10 % · proposed method 25 % · real-benchmark evaluation 25 % · stats/ablations 15 % · writing 15 %. The MVP covers the **method prototype, the controls, the statistical protocol and the evaluation harness** (≈ 20–25 %). It does **not** cover real SAM-Road runs (needs your Colab).

**Files** (in `mvp/`)
- [`roadgap_lab.py`](mvp/roadgap_lab.py) — synthetic road worlds (roads, occlusions, faint non-road distractors), families A–G, APLS-lite / buffer-F1 metrics, validation-tuned selection, bootstrap CIs.
- [`README.md`](mvp/README.md) — how to run, how to plug in real SAM-Road masks.
- `outputs/results_synthetic.txt` — the run log.

**Families compared:** A baseline threshold · B global gamma (Paper 1) · C threshold sweep (control) · D P-only "adaptive" gamma (other AI's Gap 1) · E distance-only bridging (Paper 1 A*, emulated: half-plane, `MIN_GRAPH=2×MAX`) · F gamma+E (Paper 1 hybrid, emulated) · G direction-aware + evidence-verified bridging (**ours**, with a 2×2 ablation: direction × evidence).

### Results (24 synthetic test worlds; all parameters chosen on 8 separate validation worlds; 95 % bootstrap CI of the paired difference to baseline A)

| Method | APLS-lite | Buf-P | Buf-R | Buf-F1 | Gap recall | ΔAPLS [CI] | ΔF1 [CI] |
|---|---|---|---|---|---|---|---|
| A baseline τ=0.5 | 0.623 | 0.999 | 0.897 | 0.945 | 0.167 | – | – |
| B gamma=2.0 (Paper 1) | 0.762 | 0.896 | 0.971 | 0.932 | 0.758 | +0.138 [+0.072,+0.209] | **−0.013** [−0.021,−0.005] |
| C threshold τ=0.25 (control) | 0.762 | 0.896 | 0.971 | 0.932 | 0.758 | +0.138 [+0.072,+0.209] | −0.013 [−0.021,−0.005] |
| D P-only "adaptive" gamma α=2 | 0.859 | 0.896 | 0.993 | 0.942 | 0.956 | +0.235 [+0.180,+0.294] | −0.003 [−0.010,+0.003] |
| E distance-only bridging (Paper 1 A*) | 0.861 | 0.958 | 0.940 | 0.949 | 0.550 | +0.237 [+0.182,+0.295] | +0.004 [−0.002,+0.009] |
| F gamma + E (Paper 1 hybrid) | 0.805 | 0.835 | 0.987 | 0.904 | 0.893 | +0.181 [+0.113,+0.251] | **−0.041** [−0.050,−0.032] |
| G0 evidence-only verification | 0.876 | 0.978 | 0.940 | 0.958 | 0.544 | +0.252 [+0.197,+0.311] | +0.013 [+0.008,+0.018] |
| G1 direction-only | 0.859 | 0.957 | 0.941 | 0.949 | 0.561 | +0.235 [+0.181,+0.291] | +0.004 [−0.001,+0.009] |
| **G2 direction + evidence (ours)** | **0.879** | 0.979 | 0.940 | **0.959** | 0.550 | **+0.255** [+0.201,+0.311] | **+0.014** [+0.010,+0.018] |

*APLS-lite / Buf-F1 are proxies computed on synthetic worlds I designed. They demonstrate mechanisms, not real-benchmark performance.*

### What the MVP actually shows (and doesn't)
1. **Gamma ≡ threshold — confirmed exactly.** B and C give *identical* masks and scores (`mask mismatch = 0.000000`). So Paper 1's gamma sweep is a threshold sweep in disguise unless a real-data run proves otherwise (H1 stands, needs a real-data run).
2. **The other AI's "adaptive gamma" is a threshold too — confirmed.** α=2 ⇒ exactly `P > 0.155` (0 mismatching pixels; a direct threshold run scores 0.860/0.942 vs D's 0.859/0.942). It *looked* better than B/C only because my threshold grid did not include 0.155. Do **not** present it as a spatially-adaptive method.
3. **The Paper-1-style hybrid gets worse, not better** (F: F1 −0.041). Same qualitative pattern as the paper's Table 4, where the hybrid never clearly beats its parts.
4. **Evidence verification is the effective ingredient.** G0/G2 improve APLS-lite *and* F1 with CIs excluding 0, and beat E on F1 (+0.010) at equal-or-better APLS. This is the "no spurious bridges" mechanism the report proposes.
5. **The direction prior alone did not help here** (G1 ≈ E; G2 − G0 = +0.003 APLS, negligible). ⇒ On this synthetic world the honest claim is "evidence-verified bridging"; the *direction* component is **unproven** and needs real data (real bends/curves) or a harder synthetic setup before we claim it.
6. **Trade-off visible in gap recall:** bridging recovers ~55 % of occluded road vs 76–96 % for threshold-lowering, but the latter pays with false positives (Buf-P 0.896 vs 0.979). G2 is the only family that improves APLS-lite *and* F1 significantly.
7. **Limits of this evidence:** synthetic worlds were built by me (faint distractors deliberately overlap with occluded-road probabilities), only 24 test worlds, proxies instead of official TOPO/APLS, single seed set, and G2 vs D on APLS is not statistically separable (CIs overlap). It is a **go / no-go signal for the real-data phase**, not a result.

---

## 9. Other candidate journal papers (non-conference) — seen in searches, **not** read in full
| Paper | Venue | Why relevant / caveat |
|---|---|---|
| *Occlusion-Aware Topology Refinement for Robust Road Graph Extraction from Satellite Imagery* | *Remote Sensing* 18(18):3173 (15 Sep 2026) | Closest competitor; training-based; **read this first** |
| SAM2MS: An Efficient Framework for HRSI Road Extraction Powered by SAM2 | *Remote Sensing* 17(18):3181 (Sep 2025) | SAM2 + adapters; pixel-level, training needed |
| TopoRF-Net (topology-aware road segmentation) | *Sensors* 25(24):7428 (2025) | segmentation + topology loss |
| Deep learning-based road extraction: progress, problems, perspectives | *ISPRS J. Photogramm. Remote Sens.* (2025) | high-quality survey for the related-work section |
| DOGE: Differentiable Bézier Graph Optimization | arXiv 2511.19850 | current SOTA numbers to cite (preprint) |
| SAM-Road++ (CVPR 2025), DeH4R, GLD-Road | conference / arXiv | good baselines to *cite*, not to build the "journal" claim on |

Recommendation: **Paper 1 stays the base.** SAM2-RoadNet/kLCRNet require training; Paper 2 has no topology and inconsistent numbers.

---

## 10. Suggested 4-week plan (real-data phase)
1. **Week 1:** Colab: install SAM-Road (`htcr/sam_road`), Go for APLS, download HF checkpoints; reproduce Table 1 (80.28/70.68 & 77.17/67.61). Dump `fused_road_mask`, `fused_keypoint_mask` and the output graph per test image.
2. **Week 2:** H1 controls: sweep `ROAD_THRESHOLD`, `ITSC_THRESHOLD`, `TOPO_THRESHOLD` vs γ; re-create Paper 1's Tables 2–3 with a **validation split**.
3. **Week 3:** port the MVP bridging (graph-endpoint version) + verifier; sweep on validation; freeze one configuration.
4. **Week 4:** test-set run, bootstrap CIs, occlusion-stratified analysis (tag GT-road pixels under vegetation/shadow using an RGB cue such as excess-green, or by labelling a small subset by hand), write-up.

---

## Sources
- Your PDFs: Paper 1 (JISEBI 12(1) 84–97), SAM2-RoadNet (*Remote Sens.* 18, 913), kLCRNet (JSTARS 18, 12074), U-Net (IJECES 17(5)), Review (JTTE 2016).
- SAM-Road: https://arxiv.org/abs/2403.16051 · code https://github.com/htcr/sam_road
- SAM-Road++: https://arxiv.org/abs/2411.16733 · https://github.com/earth-insights/samroadplus
- DOGE (numbers): https://arxiv.org/abs/2511.19850
- Occlusion-Aware Topology Refinement: https://www.mdpi.com/2072-4292/18/18/3173
- Crossref record of Paper 1: https://api.crossref.org/works/10.20473/jisebi.12.1.84-97 · JISEBI page https://e-journal.unair.ac.id/JISEBI/article/view/76342
- JISEBI ranking: https://si.fst.unair.ac.id/2026/04/22/jisebi-melompat-tinggi-kini-resmi-terindeks-scopus-q2/
