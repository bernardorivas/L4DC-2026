import networkx as nx
import numpy as np
import matplotlib.cm as cm
from typing import List, Set, Dict, FrozenSet

def compute_morse_graph(box_map: nx.DiGraph, assign_colors: bool = True, cmap_name: str = 'viridis') -> nx.DiGraph:
    """
    Compute the Morse graph from a BoxMap, properly handling transient states.

    The Morse graph shows connectivity between non-trivial Morse sets, where:
    1. Multi-node SCCs (recurrent components)
    2. Single-node SCCs with self-loops (fixed points)

    Connectivity includes paths through transient states (trivial SCCs).
    The resulting graph is the transitive reduction of the condensation graph
    restricted to recurrent components, which gives the Hasse diagram of the
    partial order on recurrent components.

    :param box_map: The BoxMap (directed graph), where nodes are box indices.
    :param assign_colors: If True, assign colors to Morse sets as node attributes.
    :param cmap_name: Name of matplotlib colormap to use for coloring (default: 'viridis', colorblind-friendly).
    :return: A directed graph where each node is a non-trivial Morse set,
             represented by a frozenset of the box indices it contains.
             If assign_colors=True, each node has a 'color' attribute.
    """
    # Get all strongly connected components
    all_sccs = list(nx.strongly_connected_components(box_map))
    
    # Identify non-trivial SCCs (the actual Morse sets)
    non_trivial_sccs = []
    for scc in all_sccs:
        if len(scc) > 1:
            # Multi-node SCC is always non-trivial
            non_trivial_sccs.append(scc)
        elif len(scc) == 1:
            # Single-node SCC is non-trivial only if it has a self-loop
            node = next(iter(scc))
            if box_map.has_edge(node, node):
                non_trivial_sccs.append(scc)
    
    if not non_trivial_sccs:
        # Return empty graph if no non-trivial SCCs found
        return nx.DiGraph()
    
    # Build the full condensation graph (includes all SCCs)
    condensation = nx.condensation(box_map, all_sccs)
    
    # Create mapping from SCC to condensation node
    scc_to_cnode = {}
    for i, scc in enumerate(all_sccs):
        scc_to_cnode[frozenset(scc)] = i
    
    # Convert non-trivial SCCs to frozensets
    non_trivial_frozensets = [frozenset(scc) for scc in non_trivial_sccs]
    
    # Create the Morse graph with non-trivial SCCs as nodes
    morse_graph = nx.DiGraph()
    morse_graph.add_nodes_from(non_trivial_frozensets)
    
    # Find connectivity between non-trivial SCCs through the condensation graph
    for scc1 in non_trivial_frozensets:
        for scc2 in non_trivial_frozensets:
            if scc1 != scc2:
                # Get corresponding condensation nodes
                cnode1 = scc_to_cnode[scc1]
                cnode2 = scc_to_cnode[scc2]
                
                # Check if there's a path in the condensation graph
                if nx.has_path(condensation, cnode1, cnode2):
                    morse_graph.add_edge(scc1, scc2)
    
    # Apply transitive reduction to get the Hasse diagram
    # The condensation graph is always a DAG, so the morse_graph restricted
    # to non-trivial SCCs is also a DAG (assuming recurrent components form a DAG)
    try:
        morse_graph = nx.transitive_reduction(morse_graph)
    except nx.NetworkXError:
        # If graph is not a DAG, print warning but continue
        import warnings
        warnings.warn("Morse graph is not a DAG; transitive reduction could not be applied. "
                     "This may indicate cycles in the recurrent components.")

    # Assign colors to Morse sets as node attributes
    if assign_colors:
        morse_sets = list(morse_graph.nodes())
        num_sets = len(morse_sets)
        if num_sets > 0:
            cmap = cm.get_cmap(cmap_name)
            for i, morse_set in enumerate(morse_sets):
                # Assign color as RGBA tuple
                # Note: pygraphviz may warn about RGBA tuples, but this is harmless
                # as the colors are used by matplotlib, not pygraphviz
                morse_graph.nodes[morse_set]['color'] = cmap(i / max(num_sets, 10))

    return morse_graph

def compute_all_morse_set_basins(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Compute the basin of attraction for each Morse set using reachability.

    Basin of a Morse set = all boxes in the BoxMap that eventually flow into any box in that Morse set.
    Uses efficient condensation-based algorithm for O(boxes + edges) complexity.

    When a transient box can reach multiple Morse sets, it is assigned to the basin of the
    "highest" Morse set in topological order (earliest in the DAG structure).

    NOTE: This is a fast reachability-based algorithm. For the mathematically precise
    containment-based definition, use compute_all_morse_set_basins_containment().

    :param morse_graph: The Morse graph containing all Morse sets as nodes
    :param box_map: The BoxMap (directed graph of box-to-box transitions)
    :return: Dictionary mapping each Morse set (frozenset of box indices) to its basin (set of box indices)
    """
    # Get all SCCs and build condensation
    all_sccs = list(nx.strongly_connected_components(box_map))

    # Identify non-trivial SCCs (the Morse sets)
    morse_sets_list = list(morse_graph.nodes())
    morse_sets_set = set(morse_sets_list)

    # Establish priority ranking based on topological order
    # Lower rank = higher in the DAG = takes priority for basin assignment
    morse_rank = {morse_set: rank for rank, morse_set in enumerate(nx.topological_sort(morse_graph))}

    # Create mapping from box to its SCC
    box_to_scc = {}
    for scc in all_sccs:
        scc_frozen = frozenset(scc)
        for box in scc:
            box_to_scc[box] = scc_frozen

    # Initialize basins: each Morse set contains itself
    basins = {morse_set: set(morse_set) for morse_set in morse_sets_list}

    # For each box, determine which Morse set(s) it can reach
    for box in box_map.nodes():
        box_scc = box_to_scc[box]

        # If box is already in a Morse set, skip (already in its own basin)
        if box_scc in morse_sets_set:
            continue

        # Box is transient (in trivial SCC) - find which Morse sets it can reach
        # Use BFS from this box to find reachable Morse sets
        visited = {box}
        queue = [box]
        reached_morse_sets = set()

        while queue:
            current = queue.pop(0)

            # Check if current box is in a Morse set
            current_scc = box_to_scc[current]
            if current_scc in morse_sets_set:
                reached_morse_sets.add(current_scc)
                # Don't explore beyond Morse sets
                continue

            # Explore successors
            for successor in box_map.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        # Add this box to the basin of the highest reachable Morse set (earliest in topological order)
        if reached_morse_sets:
            highest_morse_set = min(reached_morse_sets, key=lambda ms: morse_rank[ms])
            basins[highest_morse_set].add(box)

    return basins


def compute_all_morse_set_basins(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Compute disjoint basins of attraction for each Morse set using reachability.

    For each box ξ, determine which Morse sets it can reach via forward iteration,
    then assign it to the HIGHEST reachable Morse set in the partial order.

    Basin(p) = {ξ : ξ can reach M(p) and p is maximal among reachable Morse sets}

    Basins are DISJOINT: each box belongs to exactly one basin.

    :param morse_graph: The Morse graph containing all Morse sets as nodes
    :param box_map: The BoxMap (directed graph of box-to-box transitions)
    :return: Dictionary mapping each Morse set to its basin (disjoint sets of box indices)
    """
    # Step 1: For each box, compute which Morse sets it can reach
    all_boxes = set(box_map.nodes())
    box_to_reachable_morse_sets = {}

    # Get all Morse set boxes
    morse_set_to_boxes = {morse_set: set(morse_set) for morse_set in morse_graph.nodes()}
    box_to_morse_set = {}
    for morse_set in morse_graph.nodes():
        for box in morse_set:
            box_to_morse_set[box] = morse_set

    for start_box in all_boxes:
        # Forward reachability: find all Morse sets reachable from start_box
        reachable_morse_sets = set()
        visited = {start_box}
        queue = [start_box]

        while queue:
            current = queue.pop(0)
            # Check if current is in a Morse set
            if current in box_to_morse_set:
                reachable_morse_sets.add(box_to_morse_set[current])

            # Continue forward iteration
            if current in box_map:
                for successor in box_map.successors(current):
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)

        box_to_reachable_morse_sets[start_box] = reachable_morse_sets

    # Step 2: Assign each box to the HIGHEST reachable Morse set
    basins = {morse_set: set() for morse_set in morse_graph.nodes()}

    for box, reachable in box_to_reachable_morse_sets.items():
        if not reachable:
            # Box doesn't reach any Morse set (maps outside or to boundary)
            continue

        # Find highest (maximal) Morse set among reachable ones
        # A Morse set p is maximal if no other reachable Morse set q has p→q
        maximal_morse_sets = set(reachable)
        for p in reachable:
            for q in reachable:
                if p != q and nx.has_path(morse_graph, p, q):
                    # p can reach q, so p is higher than q
                    # Remove q from maximal candidates
                    maximal_morse_sets.discard(q)

        # Should have exactly one maximal element (Morse graph is a DAG/poset)
        # Assign box to the maximal Morse set
        if len(maximal_morse_sets) == 1:
            highest = maximal_morse_sets.pop()
            basins[highest].add(box)
        elif len(maximal_morse_sets) > 1:
            # Multiple maximal elements - shouldn't happen in a proper Morse graph
            # Assign to first one found
            highest = maximal_morse_sets.pop()
            basins[highest].add(box)

    return basins


def compute_all_morse_set_roas(morse_graph: nx.DiGraph, box_map: nx.DiGraph) -> Dict[FrozenSet[int], Set[int]]:
    """
    Compute the region of attraction (RoA) for each Morse set.

    TEMPORARY BANDAID: For each Morse set p, compute:
        RoA(p) = Basin(p) ∪ Basin(q) for all q reachable from p in Morse graph

    RoAs can OVERLAP: a box can belong to multiple RoAs.

    :param morse_graph: The Morse graph containing all Morse sets as nodes
    :param box_map: The BoxMap (directed graph of box-to-box transitions)
    :return: Dictionary mapping each Morse set to its RoA (overlapping sets of box indices)
    """
    # Step 1: Compute disjoint basins using reachability
    basins = compute_all_morse_set_basins(morse_graph, box_map)

    # Step 2: For each Morse set p, compute RoA(p) = union of basins of p and all successors
    roas = {}
    for morse_set_p in morse_graph.nodes():
        # Find all q reachable from p (including p itself)
        reachable_from_p = {morse_set_p}
        queue = [morse_set_p]
        visited = {morse_set_p}

        while queue:
            current = queue.pop(0)
            for successor in morse_graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
                    reachable_from_p.add(successor)

        # RoA(p) = union of basins of all q reachable from p
        roa = set()
        for q in reachable_from_p:
            roa.update(basins[q])

        roas[morse_set_p] = roa

    return roas


def full_morse_graph_analysis(grid, F, compute_basins=True):
    """
    Compute box map, morse graph, and basins for a given dynamics F.

    This is a pure computation function that performs the core Morse graph analysis
    without any visualization. 

    :param grid: UniformGrid object
    :param F: Dynamics object (F_integration, F_data, F_gaussianprocess, etc.)
    :param compute_basins: If False, skip basin computation (useful when you only need
                          basins for post-processed/combined Morse graph). Default True.
    :return: (box_map, morse_graph, basins)
        - box_map: NetworkX DiGraph of box-to-box transitions
        - morse_graph: NetworkX DiGraph of morse sets (SCCs)
        - basins: Dict mapping each morse set to its basin (set of box indices)
                  Uses fast reachability algorithm for initial Morse graphs.
                  If compute_basins=False, returns empty dict

    Example:
        >>> from MorseGraphL4DC.grids import UniformGrid
        >>> from MorseGraphL4DC.dynamics import F_integration
        >>> from MorseGraphL4DC.analysis import full_morse_graph_analysis
        >>> grid = UniformGrid(bounds=np.array([[0,0], [6,6]]), divisions=np.array([128, 128]))
        >>> F = F_integration(ode_system, tau=0.5)
        >>> box_map, morse_graph, basins = full_morse_graph_analysis(grid, F)
        
        # Skip basin computation (faster for large Morse graphs)
        >>> box_map, morse_graph, _ = full_morse_graph_analysis(grid, F, compute_basins=False)
    """
    from MorseGraphL4DC.core import Model

    # Compute box map
    model = Model(grid, F)
    box_map = model.compute_box_map()

    # Compute morse graph
    morse_graph = compute_morse_graph(box_map)

    # Compute basins using fast reachability algorithm (good for many nodes)
    if compute_basins:
        basins = compute_all_morse_set_basins(morse_graph, box_map)
    else:
        basins = {}

    return box_map, morse_graph, basins
