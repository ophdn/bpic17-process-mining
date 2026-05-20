"""
Render petri_HM.pnml and petri_IMf.pnml from output_models/ as PNG
and save them to ../figures/.
"""

import os
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_vis

# Ensure Graphviz is on PATH
GRAPHVIZ_BIN = r"C:\Program Files\Graphviz\bin"
if GRAPHVIZ_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] += f";{GRAPHVIZ_BIN}"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(SCRIPT_DIR, "output_models")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = {
    "petri_HM":  "petri_HM.pnml",
    "petri_IMf": "petri_IMf.pnml",
}

for name, filename in models.items():
    pnml_path = os.path.join(INPUT_DIR, filename)
    png_path  = os.path.join(OUTPUT_DIR, f"{name}.png")

    print(f"Loading  {pnml_path} …")
    net, im, fm = pm4py.read_pnml(pnml_path)

    print(f"Rendering {png_path} …")
    gviz = pn_vis.apply(net, im, fm)
    pn_vis.save(gviz, png_path)
    print(f"  Saved → {png_path}\n")

print("Done.")
