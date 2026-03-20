import networkx as nx
import numpy as np
from MorseGraphL4DC.utils import find_box_containing_point, compute_morse_set_centroid
from MorseGraphL4DC.analysis import compute_morse_graph, compute_all_morse_set_basins

def post_processing_example_1(grid, box_map, morse_graph, basins, T1, T2):
    """
    Post-process morse graph for example1 (toggle switch).

    Collapses the morse graph to 3 nodes:
    - Node 2: Repeller SCC containing point (T1, T2)
    - Node 0: Attractors where x1 > T1, x2 < T2
    - Node 1: Attractors where x1 < T1, x2 > T2
    - Edges: 2->0 and 2->1 (only if connectivity exists)

    RoAs are computed using the containment-based algorithm on the combined
    structure (not just merged from original basins).

    :param grid: UniformGrid object
    :param box_map: NetworkX DiGraph of box-to-box transitions
    :param morse_graph: NetworkX DiGraph of morse sets
    :param basins: Dict mapping morse sets to their basins (used for structure, not final RoAs)
    :param T1: x1 threshold parameter
    :param T2: x2 threshold parameter
    :return: (collapsed_morse_graph, combined_roas)
    """
    # Early return: if only 1 morse set, nothing to combine
    if len(morse_graph.nodes()) <= 1:
        print("  Only 1 morse set - skipping post-processing (nothing to combine)")
        return morse_graph, basins

    # Step 1: Find the repeller morse set (contains point (T1, T2))
    target_box = find_box_containing_point(grid, [T1, T2])

    repeller_node = None
    for node in morse_graph.nodes():
        if target_box in node:
            repeller_node = node
            break

    if repeller_node is None:
        raise ValueError(f"Could not find morse set containing point ({T1}, {T2})")

    # Step 2: Find all attractor morse sets (sinks)
    attractors = [node for node in morse_graph.nodes()
                  if morse_graph.out_degree(node) == 0]

    # Step 3: Identify the two terminal attractors by spatial location
    # Attractor 0: sink where x1 > T1 and x2 < T2
    # Attractor 1: sink where x1 < T1 and x2 > T2
    attractor_0 = None
    attractor_1 = None

    for attractor in attractors:
        centroid = compute_morse_set_centroid(grid, attractor)
        x1_center, x2_center = centroid[0], centroid[1]

        if x1_center > T1 and x2_center < T2:
            attractor_0 = attractor
        elif x1_center < T1 and x2_center > T2:
            attractor_1 = attractor

    # Step 3b: Classify ALL morse sets based on reachability (priority-based)
    # Priority: repeller > attractor_0 > attractor_1
    node_0_group = []  # All morse sets that flow to attractor_0 (but not repeller)
    node_1_group = []  # All morse sets that flow to attractor_1 (but not repeller)
    node_2_group = []  # All morse sets that flow to repeller (including repeller itself)

    for morse_set in morse_graph.nodes():
        # Check reachability with priority
        if morse_set == repeller_node or nx.has_path(morse_graph, morse_set, repeller_node):
            # Flows to repeller (highest priority)
            node_2_group.append(morse_set)
        elif attractor_0 is not None and (morse_set == attractor_0 or nx.has_path(morse_graph, morse_set, attractor_0)):
            # Flows to attractor_0
            node_0_group.append(morse_set)
        elif attractor_1 is not None and (morse_set == attractor_1 or nx.has_path(morse_graph, morse_set, attractor_1)):
            # Flows to attractor_1
            node_1_group.append(morse_set)

    # Step 4: Create frozensets of boxes for each combined morse set
    # Union all boxes from all morse sets in each group
    if node_0_group:
        combined_morse_set_0 = frozenset().union(*(ms for ms in node_0_group))
    else:
        combined_morse_set_0 = frozenset()

    if node_1_group:
        combined_morse_set_1 = frozenset().union(*(ms for ms in node_1_group))
    else:
        combined_morse_set_1 = frozenset()

    if node_2_group:
        combined_morse_set_2 = frozenset().union(*(ms for ms in node_2_group))
    else:
        combined_morse_set_2 = frozenset()

    collapsed_morse_set_size_0 = len(combined_morse_set_0)
    collapsed_morse_set_size_1 = len(combined_morse_set_1)
    collapsed_morse_set_size_2 = len(combined_morse_set_2)

    print(f"  Node 0: {len(node_0_group)} morse sets → {collapsed_morse_set_size_0} boxes")
    print(f"  Node 1: {len(node_1_group)} morse sets → {collapsed_morse_set_size_1} boxes")
    print(f"  Node 2: {len(node_2_group)} morse sets → {collapsed_morse_set_size_2} boxes")

    # Step 5: Build combined morse graph
    combined_morse_graph = nx.DiGraph()
    combined_morse_graph.add_nodes_from([combined_morse_set_0, combined_morse_set_1, combined_morse_set_2])

    # Check connectivity between groups based on original morse_graph edges
    # Edge 2→0: if any morse set in node_2_group has edge to any in node_0_group
    if node_0_group and node_2_group:
        for ms2 in node_2_group:
            for ms0 in node_0_group:
                if morse_graph.has_edge(ms2, ms0):
                    combined_morse_graph.add_edge(combined_morse_set_2, combined_morse_set_0)
                    break
            else:
                continue
            break

    # Edge 2→1: if any morse set in node_2_group has edge to any in node_1_group
    if node_1_group and node_2_group:
        for ms2 in node_2_group:
            for ms1 in node_1_group:
                if morse_graph.has_edge(ms2, ms1):
                    combined_morse_graph.add_edge(combined_morse_set_2, combined_morse_set_1)
                    break
            else:
                continue
            break

    # Step 6: Compute RoAs using containment-based algorithm
    # Import here to avoid circular dependency
    from MorseGraphL4DC.analysis import compute_all_morse_set_roas

    print("  Computing RoAs with containment algorithm...")
    combined_roas = compute_all_morse_set_roas(combined_morse_graph, box_map)

    # Add color attributes for visualization (Wong colorblind-safe palette)
    colors = [
        (0.0, 0.45, 0.70),   # Blue (Node 0 - Attractor 0)
        (0.90, 0.60, 0.0),   # Orange (Node 1 - Attractor 1)
        (0.80, 0.40, 0.80)   # Purple (Node 2 - Repeller)
    ]
    for morse_set, color in zip([combined_morse_set_0, combined_morse_set_1, combined_morse_set_2], colors):
        combined_morse_graph.nodes[morse_set]['color'] = color

    return combined_morse_graph, combined_roas


def post_processing_example_2(grid, box_map, morse_graph, basins):
    """
    Post-process morse graph for example2 (piecewise Van der Pol oscillator).

    Collapses the morse graph to 2 nodes:
    - Node 0: Stable limit cycle (main attractor/bottom)
    - Node 1: Unstable region near origin (repressor/top)

    RoAs are computed using the containment-based algorithm on the combined
    structure (not just merged from original basins).

    :param grid: UniformGrid object
    :param box_map: NetworkX DiGraph of box-to-box transitions
    :param morse_graph: NetworkX DiGraph of morse sets
    :param basins: Dict mapping morse sets to their basins (used for structure, not final RoAs)
    :return: (combined_morse_graph, combined_roas)
    """
    # Early return: if ≤2 morse sets, nothing to combine
    if len(morse_graph.nodes()) <= 2:
        print(f"  Only {len(morse_graph.nodes())} morse set(s) - skipping post-processing (nothing to combine)")
        return morse_graph, basins

    # Step 1: Identify "top" (repressor/source) and "bottom" (attractor/sink)
    # Top = source node (in_degree == 0, highest in topological order)
    # Bottom = sink node (out_degree == 0, lowest in topological order)

    # Find sources (repressor candidates)
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    # Find sinks (attractor candidates)
    sinks = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]

    # Take the first source as repressor, first sink as attractor
    # (if multiple, topological order will pick correctly)
    repressor = sources[0] if sources else list(morse_graph.nodes())[0]
    attractor = sinks[0] if sinks else list(morse_graph.nodes())[-1]

    # Step 2: Classify all OTHER morse sets based on spatial criterion
    # If EVERY box in morse_set is inside [-1, 1] × [-1, 1] → add to repressor group
    # Otherwise → add to attractor group

    node_0_group = [attractor]  # Attractor group (stable limit cycle)
    node_1_group = [repressor]  # Repressor group (unstable origin)

    all_boxes = grid.get_boxes()

    for morse_set in morse_graph.nodes():
        # Skip if already classified
        if morse_set == repressor or morse_set == attractor:
            continue

        # Check if ALL boxes in this morse set are inside [-1, 1] × [-1, 1]
        all_boxes_inside = True
        for box_idx in morse_set:
            if box_idx < len(all_boxes):
                box = all_boxes[box_idx]
                # Get box center
                box_center = (box[0] + box[1]) / 2
                x1_center, x2_center = box_center[0], box_center[1]

                # Check if this box is inside [-1, 1] × [-1, 1]
                if abs(x1_center) > 1.0 or abs(x2_center) > 1.0:
                    all_boxes_inside = False
                    break

        if all_boxes_inside:
            # All boxes inside switching region → repressor group
            node_1_group.append(morse_set)
        else:
            # At least one box outside → attractor group
            node_0_group.append(morse_set)

    # Step 4: Create frozensets of boxes for each combined morse set
    # Union all boxes from all morse sets in each group
    if node_0_group:
        combined_morse_set_0 = frozenset().union(*(ms for ms in node_0_group))
    else:
        combined_morse_set_0 = frozenset()

    if node_1_group:
        combined_morse_set_1 = frozenset().union(*(ms for ms in node_1_group))
    else:
        combined_morse_set_1 = frozenset()

    combined_morse_set_size_0 = len(combined_morse_set_0)
    combined_morse_set_size_1 = len(combined_morse_set_1)

    print(f"  Node 0 (Stable Limit Cycle): {len(node_0_group)} morse sets → {combined_morse_set_size_0} boxes")
    print(f"  Node 1 (Unstable Origin): {len(node_1_group)} morse sets → {combined_morse_set_size_1} boxes")

    # Step 5: Build combined morse graph
    combined_morse_graph = nx.DiGraph()
    combined_morse_graph.add_nodes_from([combined_morse_set_0, combined_morse_set_1])

    # Check connectivity between groups based on original morse_graph edges
    # Edge 1→0: if any morse set in node_1_group has edge to any in node_0_group
    if node_0_group and node_1_group:
        for ms1 in node_1_group:
            for ms0 in node_0_group:
                if morse_graph.has_edge(ms1, ms0):
                    combined_morse_graph.add_edge(combined_morse_set_1, combined_morse_set_0)
                    break
            else:
                continue
            break

    # Step 6: Compute RoAs using containment-based algorithm
    # Import here to avoid circular dependency
    from MorseGraphL4DC.analysis import compute_all_morse_set_roas

    print("  Computing RoAs with containment algorithm...")
    combined_roas = compute_all_morse_set_roas(combined_morse_graph, box_map)

    # Add color attributes for visualization (Wong colorblind-safe palette)
    colors = [
        (0.0, 0.45, 0.70),   # Blue (Node 0 - Stable Limit Cycle)
        (0.90, 0.60, 0.0)    # Orange (Node 1 - Unstable Origin)
    ]
    for morse_set, color in zip([combined_morse_set_0, combined_morse_set_1], colors):
        combined_morse_graph.nodes[morse_set]['color'] = color

    return combined_morse_graph, combined_roas