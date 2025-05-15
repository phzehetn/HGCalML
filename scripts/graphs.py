# Philipp Zehetner
# Script to create visualizations with graphviz
# that show all relevant model blocks and loss layers


# import numpy as np
# import matplotlib.pyplot as plt

from graphviz import Digraph

# Define Catppuccin-Latte color theme
CATPPUCCIN_LATTE = {
    "Rosewater": "#dc8a78",
    "Flamingo": "#dd7878",
    "Pink": "#ea76cb",
    "Mauve": "#8839ef",
    "Red": "#d20f39",
    "Maroon": "#e64553",
    "Peach": "#fe640b",
    "Yellow": "#df8e1d",
    "Green": "#40a02b",
    "Teal": "#179299",
    "Sky": "#04a5e5",
    "Sapphire": "#209fb5",
    "Blue": "#1e66f5",
    "Lavender": "#7287fd",
    "Text": "#4c4f69",
    "Subtext1": "#5c5f77",
    "Subtext0": "#6c6f85",
    "Overlay2": "#7c7f93",
    "Overlay1": "#8c8fa1",
    "Overlay0": "#9ca0b0",
    "Surface2": "#acb0be",
    "Surface1": "#bcc0cc",
    "Surface0": "#ccd0da",
    "Base": "#eff1f5",
    "Mantle": "#e6e9ef",
    "Crust": "#dce0e8",
}
THEME = CATPPUCCIN_LATTE

INPUT_STYLE = {
    "shape": "square",
    "style": "filled",
    "fillcolor": THEME["Green"],
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
FUNCTION_STYLE = {
    "shape": "octagon",
    "style": "filled",
    "fillcolor": THEME["Sky"],
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
INTERMEDIATE_STYLE = {
    "shape": "ellipse",
    "style": "filled",
    "fillcolor": THEME["Peach"],
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
BLOCK_STYLE = {
    "shape": "hexagon",
    "style": "filled",
    "fillcolor": "Yellow",
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
LOSS_STYLE = {
    "shape": "diamond",
    "style": "filled",
    "fillcolor": "orange",
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
OUTPUT_STYLE = {
    "shape": "ellipse",
    "style": "filled",
    "fillcolor": THEME["Sapphire"],
    "fontcolor": THEME["Text"],
    "fontname": "NewComputerModernSans10:bold",
    "penwidth": "2",
}
EDGE_STYLE = {
    "color": THEME["Subtext1"],
    "fontname": "NewComputerModernSans10:bold",
    "style": "solid",
    "penwidth": "2",
    "arrowhead": "vee",
}
ATTRIBUTES = {
    "fontsize": "50",
    "fontcolor": THEME["Subtext1"],
    "fontname": "NewComputerModernSans10:bold",
    "labelloc": "t",
    "labeljust": "l",
    "style": "solid",
    "color": THEME["Subtext1"],
    "penwidth": "8",
    "bgcolor": THEME["Base"],
}


def graph_ll_cluster_coordinates(path):

    dot = Digraph(engine="dot")
    dot.node("C", "Cluster Coordinates", **INPUT_STYLE)
    dot.node("SID", "Shower IDs", **INPUT_STYLE)
    dot.node("KNN", r"KNN", **FUNCTION_STYLE)
    dot.node("ND", "Distances", **INTERMEDIATE_STYLE)
    dot.node("AvgPos", "Shower Centre", **INTERMEDIATE_STYLE)
    dot.node("DtS", "Distance to \nCorrect Shower Centre", **INTERMEDIATE_STYLE)
    dot.node("DtO", "Distance to \nOther Shower Centres", **INTERMEDIATE_STYLE)
    dot.node("ASF", r"Attractive Scaling Function", **FUNCTION_STYLE)
    dot.node("RSF", r"Repulsive Scaling Function", **FUNCTION_STYLE)
    dot.node("Loss", r"Loss", **OUTPUT_STYLE)

    dot.edge("C", "KNN", **EDGE_STYLE)
    dot.edge("KNN", "ND", **EDGE_STYLE)
    dot.edge("C", "AvgPos", **EDGE_STYLE)
    dot.edge("SID", "AvgPos", **EDGE_STYLE)
    dot.edge("ND", "DtS", **EDGE_STYLE)
    dot.edge("ND", "DtO", **EDGE_STYLE)
    dot.edge("AvgPos", "DtS", **EDGE_STYLE)
    dot.edge("AvgPos", "DtO", **EDGE_STYLE)
    dot.edge("DtS", "ASF", **EDGE_STYLE)
    dot.edge("DtO", "RSF", **EDGE_STYLE)
    dot.edge("ASF", "Loss", **EDGE_STYLE)
    dot.edge("RSF", "Loss", **EDGE_STYLE)

    dot.attr(label="LL_ClusterCoordinates", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)

    return


def graph_ll_oc_base(path):
    dot = Digraph(engine="dot")
    # TODO: Continue here tomorrow

    dot.attr(label="LL_OC_base", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_ll_graph_condensation_loss(path):
    dot = Digraph(engine="dot")
    dot.node("S", r"Score $s_\text{V,K}$", **INPUT_STYLE)
    dot.node("C", "Cluster Coordinates", **INPUT_STYLE)
    dot.node("SID", "Shower IDs", **INPUT_STYLE)
    dot.node("E", "Shower Energy", **INPUT_STYLE)
    dot.node("KNN", r"KNN", **FUNCTION_STYLE)
    dot.node("NID", "Neighbour Relationship", **INTERMEDIATE_STYLE)
    dot.node("N_Score", "Neighbour Scores", **INTERMEDIATE_STYLE)
    dot.node("N_SID", "Neigbour Shower ID", **INTERMEDIATE_STYLE)
    dot.node(
        "is_same",
        "Mask: \nNoise or no neighbour \nfrom same shower",
        **INTERMEDIATE_STYLE
    )
    dot.node("N_dist", "Neighbour Distances", **INTERMEDIATE_STYLE)
    dot.node(
        "MMN", "Highest Score of Neighbour\nfrom same shower", **INTERMEDIATE_STYLE
    )
    dot.node("MNS", "Neighbour with Highest Score", **INTERMEDIATE_STYLE)
    dot.node("DS", r"Distance Scaling", **FUNCTION_STYLE)
    dot.node("DW", r"Distance Weights $w_\text{d}$", **INTERMEDIATE_STYLE)
    dot.node("L0", "Neighbour with highest score:\n", **INTERMEDIATE_STYLE)
    dot.node("L1", "Other Neighbours:\n", **INTERMEDIATE_STYLE)
    dot.node(
        "L",
        "Sum over Neighbours, \nnormalized by entries \nleft by the mask",
        **OUTPUT_STYLE
    )

    dot.edge("C", "KNN", **EDGE_STYLE)
    dot.edge("KNN", "NID", **EDGE_STYLE)
    dot.edge("KNN", "N_dist", **EDGE_STYLE)
    dot.edge("S", "N_Score", **EDGE_STYLE)
    dot.edge("NID", "N_Score", **EDGE_STYLE)
    dot.edge("SID", "N_SID", **EDGE_STYLE)
    dot.edge("NID", "N_SID", **EDGE_STYLE)
    dot.edge("N_SID", "is_same", **EDGE_STYLE)
    dot.edge("SID", "is_same", **EDGE_STYLE)
    dot.edge("MMN", "MNS", **EDGE_STYLE)
    dot.edge("is_same", "MNS", **EDGE_STYLE)
    dot.edge("N_Score", "MMN", **EDGE_STYLE)
    dot.edge("N_SID", "MMN", **EDGE_STYLE)
    dot.edge("is_same", "DS", **EDGE_STYLE)
    dot.edge("N_dist", "DS", **EDGE_STYLE)
    dot.edge("MNS", "DS", **EDGE_STYLE)
    dot.edge("DS", "DW", **EDGE_STYLE)
    dot.edge("DW", "L0", **EDGE_STYLE)
    dot.edge("DW", "L1", **EDGE_STYLE)
    dot.edge("S", "L0", **EDGE_STYLE)
    dot.edge("S", "L1", **EDGE_STYLE)
    dot.edge("L0", "L", **EDGE_STYLE)
    dot.edge("L1", "L", **EDGE_STYLE)

    dot.attr(label="Graph Condensation Loss", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_gravnet(path):
    dot = Digraph(engine="dot")
    dot.node("I", "Input Features", **INPUT_STYLE)
    dot.node("O", "Output Features", **OUTPUT_STYLE)
    dot.node("D1", r"Dense Layer\nN=4", **FUNCTION_STYLE)
    dot.node("D2", r"Dense Layer\nN=64", **FUNCTION_STYLE)
    dot.node("D3", r"Dense Layer\nN=64", **FUNCTION_STYLE)
    dot.node("Coord", "Coordinates", **INTERMEDIATE_STYLE)
    dot.node("KNN", r"K-Nearest-Neighbour", **FUNCTION_STYLE)
    dot.node("Ns", "Neighbour Indices", **INTERMEDIATE_STYLE)
    dot.node("Dists", "Distances", **INTERMEDIATE_STYLE)
    dot.node(
        "MM",
        "Distance Weighted\n(Mean & Maximum Values)\n- Features",
        **INTERMEDIATE_STYLE
    )
    dot.node("Concat", "Concatenate", **FUNCTION_STYLE)

    dot.edge("I", "D1", **EDGE_STYLE)
    dot.edge("D1", "Coord", **EDGE_STYLE)
    dot.edge("Coord", "KNN", **EDGE_STYLE)
    dot.edge("KNN", "Dists", **EDGE_STYLE)
    dot.edge("KNN", "Ns", **EDGE_STYLE)
    dot.edge("Ns", "MM", **EDGE_STYLE)
    dot.edge("Dists", "MM", **EDGE_STYLE)
    dot.edge("I", "D2", **EDGE_STYLE)
    dot.edge("D2", "MM", **EDGE_STYLE)
    dot.edge("I", "Concat", **EDGE_STYLE)
    dot.edge("MM", "Concat", **EDGE_STYLE)
    dot.edge("Concat", "D3", **EDGE_STYLE)
    dot.edge("D3", "O", **EDGE_STYLE)
    dot.attr(label="GravNet", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_translation_equivariant_message_passing(path):
    dot = Digraph(engine="dot")

    dot.node("I", "Input Features", **INPUT_STYLE)
    dot.node("Ones", "Ones", **INTERMEDIATE_STYLE)
    dot.node("Nid", "Neighbour Indices", **INTERMEDIATE_STYLE)
    dot.node("Ndist", "Neighbour Distances", **INTERMEDIATE_STYLE)
    dot.node("LN", r"Layer Normalization", **FUNCTION_STYLE)
    dot.node("dm", r"d → exp(-10 d²)", **FUNCTION_STYLE)
    dot.node("MM", "Distance Weighted\nMean Values · K", **FUNCTION_STYLE)
    dot.node(
        "S1",
        r"1st Iteration: \nSubtract weighted Ones\nfrom Weighted Features\nDivide by K",
        **FUNCTION_STYLE
    )
    dot.node("S2", r"Subsequent Iterations:\n\n\nDivide by K", **FUNCTION_STYLE)
    dot.node("D1", r"Dense Layer\nN=64", **FUNCTION_STYLE)
    dot.node("O", "Output", **OUTPUT_STYLE)

    dot.edge("I", "LN", **EDGE_STYLE)
    dot.edge("Ndist", "dm", **EDGE_STYLE)
    dot.edge("dm", "MM", **EDGE_STYLE)
    dot.edge("LN", "MM", **EDGE_STYLE)
    dot.edge("Nid", "MM", **EDGE_STYLE)
    dot.edge("Ones", "MM", **EDGE_STYLE)
    dot.edge("MM", "S1", **EDGE_STYLE)
    dot.edge("MM", "S2", **EDGE_STYLE)
    dot.edge("S1", "D1", **EDGE_STYLE)
    dot.edge("S2", "D1", **EDGE_STYLE)
    dot.edge("D1", "I", **EDGE_STYLE)
    dot.edge("D1", "O", **EDGE_STYLE)
    dot.attr(label="Translation Equivariant\nMessage Passing", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_gravnet_teqmp(path):
    dot = Digraph(engine="dot")

    dot.node("I", "Input Features", **INPUT_STYLE)
    dot.node("GN", r"GravNet\nBlock", **FUNCTION_STYLE)
    dot.node("XGN", "Features", INTERMEDIATE_STYLE)
    dot.node("GN_coord", "Coordinates", **INTERMEDIATE_STYLE)
    dot.node("Nid", "Neighbour IDs", **INTERMEDIATE_STYLE)
    dot.node("Ndist", "Neighbour Distances", **INTERMEDIATE_STYLE)
    dot.node("Concat", r"Concatenate", **FUNCTION_STYLE)
    dot.node("D1", "Dense\nN=1\nNo Activation", **FUNCTION_STYLE)
    dot.node("Sdist", "Distance \nScaling", **FUNCTION_STYLE)
    dot.node("TEMP", "Translation Equivarian\nMessage Passing", **FUNCTION_STYLE)
    dot.node("Concat2", r"Concatenate", **FUNCTION_STYLE)
    dot.node("O", "Output", **OUTPUT_STYLE)

    dot.edge("I", "GN", **EDGE_STYLE)
    dot.edge("GN", "XGN", **EDGE_STYLE)
    dot.edge("GN", "GN_coord", **EDGE_STYLE)
    dot.edge("GN", "Nid", **EDGE_STYLE)
    dot.edge("GN", "Ndist", **EDGE_STYLE)
    dot.edge("XGN", "Concat", **EDGE_STYLE)
    dot.edge("I", "Concat", **EDGE_STYLE)
    dot.edge("Concat", "D1", **EDGE_STYLE)
    dot.edge("D1", "Sdist", **EDGE_STYLE)
    dot.edge("Ndist", "Sdist", **EDGE_STYLE)
    dot.edge("Sdist", "TEMP", **EDGE_STYLE)
    dot.edge("XGN", "TEMP", **EDGE_STYLE)
    dot.edge("Nid", "TEMP", **EDGE_STYLE)
    dot.edge("TEMP", "Concat2", **EDGE_STYLE)
    dot.edge("XGN", "Concat2", **EDGE_STYLE)
    dot.edge("Concat2", "O", **EDGE_STYLE)

    dot.attr(label="GraveNet + Message Passing", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_create_graph_condensation(path):
    dot = Digraph(comment="Styled Neural Network Model Block", engine="dot")
    # dot.attr(rankdir='LR')  # Set graph orientation to Left-to-Right

    # Define nodes with custom styles
    dot.node("S", "Score\nBetween 0, 1", **INPUT_STYLE)
    dot.node("C", "Coordinates", **INPUT_STYLE)
    dot.node("T", "Threshold\nInitially 0.5", **INPUT_STYLE)
    dot.node(
        "Class",
        "Classes\n · Be Neighbour        \n · Have Neighbours     \n · Always promote (tracks)",
        **FUNCTION_STYLE
    )
    dot.node(
        "AdaptThresh", "Adapt Threshold\nto not lose\nany shower", **FUNCTION_STYLE
    )
    dot.node("KNN", "Directional \nKNN", **FUNCTION_STYLE)
    dot.node("GC", "Graph\nCondensation", **OUTPUT_STYLE)

    dot.edge("S", "AdaptThresh", **EDGE_STYLE)
    dot.edge("T", "AdaptThresh", **EDGE_STYLE)
    dot.edge("C", "KNN", **EDGE_STYLE)
    dot.edge("AdaptThresh", "Class", **EDGE_STYLE)
    dot.edge("S", "Class", **EDGE_STYLE)
    dot.edge("Class", "KNN", **EDGE_STYLE)
    dot.edge("KNN", "GC", **EDGE_STYLE)

    dot.attr(label="Create Graph Condensation", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_mini_tree_create(path):
    dot = Digraph(engine="dot")

    dot.node("S", "Score (1D)", **INPUT_STYLE)
    dot.node("C", "Coordinates", **INPUT_STYLE)
    dot.node("SID", "Shower IDs", **INPUT_STYLE)
    dot.node("E", "Energy", **INPUT_STYLE)
    dot.node("IT", "Is Track?", **INPUT_STYLE)
    dot.node("LLCluster", "Clustering\nLoss", **LOSS_STYLE)
    dot.node("LLGraph", "Graph\nCondensation\nScore", **LOSS_STYLE)
    dot.node("GraphMetric", "Graph\nCondensation\nMetric", **LOSS_STYLE)
    dot.node("S0", "Score\n0 for Tracks", **INTERMEDIATE_STYLE)
    dot.node("CGC", "Create\nGraphCondensation", **BLOCK_STYLE)
    dot.node("Graph", "Output Graph", **OUTPUT_STYLE)

    dot.edge("S", "LLCluster", **EDGE_STYLE)
    dot.edge("C", "LLCluster", **EDGE_STYLE)
    dot.edge("SID", "LLCluster", **EDGE_STYLE)
    dot.edge("S", "S0", **EDGE_STYLE)
    dot.edge("IT", "S0", **EDGE_STYLE)
    dot.edge("S0", "LLGraph", **EDGE_STYLE)
    dot.edge("C", "LLGraph", **EDGE_STYLE)
    dot.edge("SID", "LLGraph", **EDGE_STYLE)
    dot.edge("E", "LLGraph", **EDGE_STYLE)
    dot.edge("S", "CGC", **EDGE_STYLE)
    dot.edge("C", "CGC", **EDGE_STYLE)
    dot.edge("CGC", "Graph", **EDGE_STYLE)
    dot.edge("Graph", "GraphMetric", **EDGE_STYLE)
    dot.attr(label="Mini Tree Create", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_mini_tree_create_simplified(path):
    dot = Digraph(engine="dot")

    dot.node("S", "Score (1D)", **INPUT_STYLE)
    dot.node("C", "Coordinates", **INPUT_STYLE)
    dot.node("SID", "Shower IDs", **INPUT_STYLE)
    dot.node("E", "Energy", **INPUT_STYLE)
    dot.node("LLCluster", "Clustering\nLoss", **LOSS_STYLE)
    dot.node("LLGraph", "Graph\nCondensation\nScore", **LOSS_STYLE)
    dot.node("GraphMetric", "Graph\nCondensation\nMetric", **LOSS_STYLE)
    dot.node("CGC", "Create\nGraphCondensation", **BLOCK_STYLE)
    dot.node("Graph", "Output Graph", **OUTPUT_STYLE)

    dot.edge("S", "LLCluster", **EDGE_STYLE)
    dot.edge("C", "LLCluster", **EDGE_STYLE)
    dot.edge("SID", "LLCluster", **EDGE_STYLE)
    dot.edge("S", "LLGraph", **EDGE_STYLE)
    dot.edge("C", "LLGraph", **EDGE_STYLE)
    dot.edge("SID", "LLGraph", **EDGE_STYLE)
    dot.edge("E", "LLGraph", **EDGE_STYLE)
    dot.edge("S", "CGC", **EDGE_STYLE)
    dot.edge("C", "CGC", **EDGE_STYLE)
    dot.edge("CGC", "Graph", **EDGE_STYLE)
    dot.edge("Graph", "GraphMetric", **EDGE_STYLE)

    dot.attr(label="Mini Tree Create", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_mini_tree_cluster(path):
    dot = Digraph(engine="dot")

    dot.node("IGC", "Input:\nGraph\nCondensation", **INPUT_STYLE)
    dot.node("D0", "Dense\nN=32\nElu:", **FUNCTION_STYLE)
    dot.node("F", "Features", INTERMEDIATE_STYLE)
    dot.node("ND", "Features:\nNeighbour\nDifference", **INTERMEDIATE_STYLE)
    dot.node("D1-3", "3xDense\nN=64\nElu", **FUNCTION_STYLE)
    dot.node("DS", "Dense\nN=K+1\nSigmoid", **FUNCTION_STYLE)
    dot.node("LL", "Edges\nLoss\nLayer", **LOSS_STYLE)
    dot.node("UGC", "Updated\nGraph\nCondensation", **OUTPUT_STYLE)
    dot.node("AGG", "Aggregate Features\nto\nCondensation Points", **OUTPUT_STYLE)

    dot.edge("IGC", "F", **EDGE_STYLE)
    dot.edge("F", "D0", **EDGE_STYLE)
    dot.edge("D0", "ND", **EDGE_STYLE)
    dot.edge("IGC", "ND", **EDGE_STYLE)
    dot.edge("ND", "D1-3", **EDGE_STYLE)
    dot.edge("D1-3", "DS", **EDGE_STYLE)
    dot.edge("DS", "LL", **EDGE_STYLE)
    dot.edge("DS", "UGC", **EDGE_STYLE)
    dot.edge("UGC", "AGG", **EDGE_STYLE)

    dot.attr(label="Mini Tree Cluster", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_tree_condensation_block_v0(path):
    dot = Digraph(engine="dot")
    dot.node("F", "Features", **INPUT_STYLE)
    dot.node("C", "Coordinates", **INPUT_STYLE)
    dot.node("E", "Energy", **INPUT_STYLE)
    dot.node("SID", "Shower ID", **INPUT_STYLE)
    dot.node("D0", "Dense\nN=128\ntanh", **FUNCTION_STYLE)
    dot.node("Concat0", "Concatenate", **FUNCTION_STYLE)
    dot.node("X0", "X'", INTERMEDIATE_STYLE)
    dot.node("TEQMP", "TEQMP", **BLOCK_STYLE)
    dot.node("X1", "X'", **INTERMEDIATE_STYLE)
    dot.node("X2", "X''", **INTERMEDIATE_STYLE)
    dot.node("C1", "Coordinates'", **INTERMEDIATE_STYLE)
    dot.node("Concat1", "Concatenate", **FUNCTION_STYLE)
    dot.node("DS", "Dense\nN=1\nSigmoid", **FUNCTION_STYLE)
    dot.node("Score", "Score", **INTERMEDIATE_STYLE)
    dot.node("TreeCreate", "Create\nTree", **BLOCK_STYLE)
    dot.node("TreeCluster", "Cluster\nTree", **BLOCK_STYLE)
    dot.node("G0", "Graph", **OUTPUT_STYLE)
    dot.node("G1", "Output\nGraph", **OUTPUT_STYLE)

    dot.edge("F", "D0", **EDGE_STYLE)
    dot.edge("D0", "Concat0", **EDGE_STYLE)
    dot.edge("C", "Concat0", **EDGE_STYLE)
    dot.edge("Concat0", "X0", **EDGE_STYLE)
    dot.edge("X0", "TEQMP", **EDGE_STYLE)
    dot.edge("C", "TEQMP", **EDGE_STYLE)
    dot.edge("E", "TEQMP", **EDGE_STYLE)
    dot.edge("SID", "TEQMP", **EDGE_STYLE)
    dot.edge("TEQMP", "X1", **EDGE_STYLE)
    dot.edge("TEQMP", "C1", **EDGE_STYLE)
    dot.edge("X0", "Concat1", **EDGE_STYLE)
    dot.edge("X1", "Concat1", **EDGE_STYLE)
    dot.edge("Concat1", "X2", **EDGE_STYLE)
    dot.edge("X2", "DS", **EDGE_STYLE)
    dot.edge("DS", "Score", **EDGE_STYLE)
    dot.edge("Score", "TreeCreate", **EDGE_STYLE)
    dot.edge("C1", "TreeCreate", **EDGE_STYLE)
    dot.edge("SID", "TreeCreate", **EDGE_STYLE)
    dot.edge("E", "TreeCreate", **EDGE_STYLE)
    dot.edge("TreeCreate", "G0", **EDGE_STYLE)
    dot.edge("G0", "TreeCluster", **EDGE_STYLE)
    dot.edge("X2", "TreeCluster", **EDGE_STYLE)
    dot.edge("TreeCluster", "G1", **EDGE_STYLE)

    dot.attr(label="TreeCondensationBlock", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_block_tree_condensation_block(path):
    dot = Digraph(engine="dot")

    dot.node("PC", "Prime Coordinates", **INPUT_STYLE)
    dot.node("IT", "Is Track", **INPUT_STYLE)
    dot.node("X", "Features", **INPUT_STYLE)
    dot.node("E", "Energy", **INPUT_STYLE)
    dot.node("SID", "Shower ID", **INPUT_STYLE)

    dot.node("D0", "Dense\nN=32\nActivation: tanh", **FUNCTION_STYLE)
    dot.node("Concat0", "Concatenate", **FUNCTION_STYLE)
    dot.node(
        "GN+TEQMP", "GravNet + Translation Equivariant\nMessage Passing", **BLOCK_STYLE
    )
    dot.node("XGN", "GravNet Features", **INTERMEDIATE_STYLE)
    dot.node("CGN", "GravNet Coordinates", **INTERMEDIATE_STYLE)
    dot.node("Concat1", "Concatenate", **FUNCTION_STYLE)
    dot.node("DScore", "Dense\nN=1\nActivation: Sigmoid", **FUNCTION_STYLE)

    dot.node("MTC", "Mini Tree Create", **BLOCK_STYLE)
    dot.node("graph", "Graph", **INTERMEDIATE_STYLE)
    dot.node("MTCl", "Mini Tree Cluster", **BLOCK_STYLE)
    dot.node("output", "Output", **OUTPUT_STYLE)

    dot.edge("X", "D0", **EDGE_STYLE)
    dot.edge("D0", "Concat0", **EDGE_STYLE)
    dot.edge("PC", "Concat0", **EDGE_STYLE)
    dot.edge("Concat0", "GN+TEQMP", **EDGE_STYLE)
    dot.edge("PC", "GN+TEQMP", **EDGE_STYLE)
    dot.edge("E", "GN+TEQMP", **EDGE_STYLE)
    dot.edge("SID", "GN+TEQMP", **EDGE_STYLE)
    dot.edge("GN+TEQMP", "XGN", **EDGE_STYLE)
    dot.edge("GN+TEQMP", "CGN", **EDGE_STYLE)
    dot.edge("Concat0", "Concat1", **EDGE_STYLE)
    dot.edge("XGN", "Concat1", **EDGE_STYLE)
    dot.edge("Concat1", "DScore", **EDGE_STYLE)
    dot.edge("CGN", "MTC", **EDGE_STYLE)
    dot.edge("DScore", "MTC", **EDGE_STYLE)
    dot.edge("MTC", "graph", **EDGE_STYLE)
    dot.edge("graph", "MTCl", **EDGE_STYLE)
    dot.edge("X", "MTCl", **EDGE_STYLE)
    dot.edge("MTCl", "output", **EDGE_STYLE)

    dot.attr(label="TreeCondensationBlock", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_snowflake_model_simple(path):
    dot = Digraph(engine="dot")
    dot.node("I", "Input Features", **INPUT_STYLE)
    dot.node("Pre", "Preprocessing", **BLOCK_STYLE)
    dot.node("DTC", "Double Tree Condensation Block", **BLOCK_STYLE)
    dot.node("CO", "Create Outputs", **BLOCK_STYLE)
    dot.node("OCL", "Object Condensation Loss", **BLOCK_STYLE)

    dot.edge("I", "Pre", **EDGE_STYLE)
    dot.edge("Pre", "DTC", **EDGE_STYLE)
    dot.edge("DTC", "CO", **EDGE_STYLE)
    dot.edge("CO", "OCL", **EDGE_STYLE)

    dot.attr(label="Full Model", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_double_tree_condensation_block(path):
    dot = Digraph(engine="dot")

    dot.node("X", "Features", **INPUT_STYLE)
    dot.node("PC", "Prime Coordinates", **INPUT_STYLE)
    dot.node("KNN", "KNN K=16", **FUNCTION_STYLE)
    dot.node("Ns", "Neighbour IDs", **INTERMEDIATE_STYLE)
    dot.node("Ds", "Neighbour Distances", **INTERMEDIATE_STYLE)
    dot.node("Concat0", "Concatenate", **FUNCTION_STYLE)
    dot.node("D0", "Dense\nN=16\nActivation: tanh", **FUNCTION_STYLE)
    dot.node("DScale", "Dense\nN=1\nno activation", **FUNCTION_STYLE)
    dot.node("LDS", "Local Distance Scaling", **BLOCK_STYLE)
    dot.node("TIMP", "Translation Invariant Message Passing", **BLOCK_STYLE)
    dot.node("Concat1", "Concatenate", **FUNCTION_STYLE)
    dot.node("TCB", "Tree Condensation Block", **BLOCK_STYLE)
    # Ouput from Tree Condensation Block: out, graph, x_proc

    dot.edge("PC", "KNN", **EDGE_STYLE)
    dot.edge("KNN", "Ns", **EDGE_STYLE)
    dot.edge("KNN", "Ds", **EDGE_STYLE)
    dot.edge("X", "Concat0", **EDGE_STYLE)
    dot.edge("PC", "Concat0", **EDGE_STYLE)
    dot.edge("Concat0", "D0", **EDGE_STYLE)
    dot.edge("D0", "DScale", **EDGE_STYLE)
    dot.edge("DScale", "LDS", **EDGE_STYLE)
    dot.edge("Ds", "LDS", **EDGE_STYLE)
    dot.edge("LDS", "TIMS", **EDGE_STYLE)
    dot.edge("D0", "TIMS", **EDGE_STYLE)
    dot.edge("Ns", "TIMS", **EDGE_STYLE)
    dot.edge("X", "Concat1", **EDGE_STYLE)
    dot.edge("TIMS", "Concat1", **EDGE_STYLE)

    dot.attr(label="Full Model", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_full_model(path):
    dot = Digraph(engine="dot")
    dot.node("I", "Input\nFeatures", **INPUT_STYLE)
    dot.node("C", "Input\nCoordinates", **INPUT_STYLE)
    dot.node("PC", "Prime\nCoordinates", **INPUT_STYLE)
    dot.node("Concat0", "Concatenate", **FUNCTION_STYLE)
    dot.node("all_features", "All Features", **INTERMEDIATE_STYLE)
    dot.node("D1", "Dense Layer\nN=64\nActivation: Elu", **FUNCTION_STYLE)
    dot.node("D2", "Dense Layer\nN=64\nActivation: Elu", **FUNCTION_STYLE)
    dot.node("D3", "Dense Layer\nN=64\nActivation: Elu", **FUNCTION_STYLE)
    dot.node("Concat1", "Concatenate", **FUNCTION_STYLE)
    dot.node("BN1", "Batch Normalization", **FUNCTION_STYLE)
    dot.node("GN_TEQMP", "GravNet + TEQMP", **BLOCK_STYLE)
    dot.node("BN2", "Batch Normalization", **FUNCTION_STYLE)
    dot.node("D4", "Dense Layer\nN=128", **FUNCTION_STYLE)
    dot.node("D5", "Dense Layer\nN=128", **FUNCTION_STYLE)
    dot.node("D6", "Dense Layer\nN=64", **FUNCTION_STYLE)
    dot.node("BN3", "Batch Normalization", **FUNCTION_STYLE)
    dot.node("CO", "Create Outputs", **BLOCK_STYLE)
    dot.node("Loss", "Loss Layer", **LOSS_STYLE)

    dot.edge("I", "Concat0", **EDGE_STYLE)
    dot.edge("C", "Concat0", **EDGE_STYLE)
    dot.edge("Concat0", "all_features", **EDGE_STYLE)
    dot.edge("I", "D1", **EDGE_STYLE)
    dot.edge("D1", "D2", **EDGE_STYLE)
    dot.edge("D2", "D3", **EDGE_STYLE)
    dot.edge("PC", "Concat1", **EDGE_STYLE)
    dot.edge("D3", "Concat1", **EDGE_STYLE)
    dot.edge("Concat1", "BN1", **EDGE_STYLE)
    dot.edge("BN1", "GN_TEQMP", **EDGE_STYLE)
    dot.edge("GN_TEQMP", "BN2", **EDGE_STYLE)
    dot.edge("BN2", "Concat0", **EDGE_STYLE)
    dot.edge("BN2", "D1", label="Repeat 3x", **EDGE_STYLE)

    dot.edge("all_features", "D4", **EDGE_STYLE)
    dot.edge("D4", "D5", **EDGE_STYLE)
    dot.edge("D5", "D6", **EDGE_STYLE)
    dot.edge("D6", "BN3", **EDGE_STYLE)
    dot.edge("BN3", "CO", **EDGE_STYLE)
    dot.edge("CO", "Loss", **EDGE_STYLE)

    dot.attr(label="Full Model", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


def graph_full_model_simple(path):
    dot = Digraph(engine="dot")
    dot.node("I", "Input\nFeatures", **INPUT_STYLE)
    dot.node("C", "Input\nCoordinates", **INPUT_STYLE)
    dot.node("PC", "Prime\nCoordinates", **INPUT_STYLE)
    dot.node("Concat0", "Concatenate", **FUNCTION_STYLE)
    dot.node("D1-3", "3x Dense Layer\nN=64\nActivation: Elu", **FUNCTION_STYLE)
    dot.node("Concat1", "Concatenate", **FUNCTION_STYLE)
    dot.node("BN1", "Batch Normalization", **FUNCTION_STYLE)
    dot.node("GN_TEQMP", "GravNet + TEQMP", **BLOCK_STYLE)
    dot.node("BN2", "Batch Normalization", **FUNCTION_STYLE)
    dot.node(
        "D4-6", "3x Dense Layer\nN=128, 128, 64\nActivation: Elu", **FUNCTION_STYLE
    )
    dot.node("BN3", "Batch Normalization", **FUNCTION_STYLE)
    dot.node("CO", "Create Outputs", **BLOCK_STYLE)
    dot.node("Loss", "Loss Layer", **LOSS_STYLE)

    dot.edge("I", "Concat0", **EDGE_STYLE)
    dot.edge("C", "Concat0", **EDGE_STYLE)
    dot.edge("I", "D1-3", **EDGE_STYLE)
    dot.edge("PC", "Concat1", **EDGE_STYLE)
    dot.edge("D1-3", "Concat1", **EDGE_STYLE)
    dot.edge("Concat1", "BN1", **EDGE_STYLE)
    dot.edge("BN1", "GN_TEQMP", **EDGE_STYLE)
    dot.edge("GN_TEQMP", "BN2", **EDGE_STYLE)
    dot.edge("BN2", "Concat0", **EDGE_STYLE)
    dot.edge("BN2", "D1-3", label="Repeat 3x", **EDGE_STYLE)

    dot.edge("Concat0", "D4-6", **EDGE_STYLE)
    dot.edge("D4-6", "BN3", **EDGE_STYLE)
    dot.edge("BN3", "CO", **EDGE_STYLE)
    dot.edge("CO", "Loss", **EDGE_STYLE)

    dot.attr(label="Full Model", **ATTRIBUTES)
    dot.render(path, format="svg", cleanup=True)
    return


if __name__ == "__main__":
    import os

    PLOTDIR = "plots"
    # Loss Layers
    graph_ll_cluster_coordinates(os.path.join(PLOTDIR, "LL_ClusterCoordinates"))
    graph_ll_graph_condensation_loss(os.path.join(PLOTDIR, "LL_GraphCondensation"))
    graph_ll_oc_base(os.path.join(PLOTDIR, "OC_Loss"))
    # Model blocks - GravNet & TEQMP
    graph_block_gravnet(os.path.join(PLOTDIR, "GravNet"))
    graph_block_translation_equivariant_message_passing(os.path.join(PLOTDIR, "TEQMP"))
    graph_block_gravnet_teqmp(os.path.join(PLOTDIR, "GravNet"))
    # Model blocks - Snowflake
    graph_block_create_graph_condensation(
        os.path.join(PLOTDIR, "Create_Graph_Condensation")
    )
    graph_block_tree_condensation_block(os.path.join(PLOTDIR, "TreeCondensationBlock"))
    graph_mini_tree_create(os.path.join(PLOTDIR, "MiniTreeCreate"))
    graph_mini_tree_cluster(os.path.join(PLOTDIR, "MiniTreeCluster"))
    # Full models
    graph_snowflake_model_simple(os.path.join(PLOTDIR, "snoflake_model_simple"))
    graph_full_model(os.path.join(PLOTDIR, "full_model"))
    graph_full_model_simple(os.path.join(PLOTDIR, "full_model_simple"))
