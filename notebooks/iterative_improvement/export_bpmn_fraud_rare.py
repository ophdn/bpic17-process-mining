"""
export_bpmn_fraud_rare.py — Export the best (FINAL_V3_IMf_noise0.5) model as BPMN 2.0
===========================================================================
Reads the existing Petri net (petri_FINAL_V3_IMf_noise0.5.pnml) from the output folder,
converts it to BPMN 2.0, and saves it as:

    output/bpmn_FINAL_V3_IMf_noise0.5.bpmn

Run from any directory:
    python iterative_improvement/export_bpmn_fraud_rare.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pm4py
import common

PNML_PATH = os.path.join(common.OUTPUT_DIR, "petri_FINAL_V3_IMf_noise0.5.pnml")
BPMN_PATH = os.path.join(common.OUTPUT_DIR, "bpmn_FINAL_V3_IMf_noise0.5.bpmn")

print(f"Loading Petri net from: {PNML_PATH}")
net, im, fm = pm4py.read_pnml(PNML_PATH)
print(f"  Places: {len(net.places)}, Transitions: {len(net.transitions)}, Arcs: {len(net.arcs)}")

print("Converting to BPMN 2.0 ...")
bpmn = pm4py.convert_to_bpmn(net, im, fm)

pm4py.write_bpmn(bpmn, BPMN_PATH)
print(f"  [saved] {BPMN_PATH}")

# ── PNG export ────────────────────────────────────────────────────────────────
print("Rendering BPMN as PNG ...")
from pm4py.visualization.bpmn import visualizer as bpmn_vis

PNG_PATH = os.path.join(common.OUTPUT_DIR, "bpmn_FINAL_V3_IMf_noise0.5.png")
gviz = bpmn_vis.apply(bpmn)
bpmn_vis.save(gviz, PNG_PATH)
print(f"  [saved] {PNG_PATH}")
print("Done.")
