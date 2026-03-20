"""
Metrics for comparing Morse graphs and their basins of attraction.

This module provides functions to evaluate the accuracy and topological similarity
of computed Morse graphs against ground truth, including:
- Region of Attraction (RoA) accuracy via Intersection over Union (IoU)
- Graph structure comparison via edit distance and isomorphism
- Algebraic topology comparison via Betti numbers
"""

import networkx as nx
import numpy as np
from typing import Dict, Set, FrozenSet, Tuple, List, Optional
from MorseGraphL4DC.utils import compute_morse_set_centroid


def match_morse_sets_by_topology(morse_graph_1: nx.DiGraph,
                                  morse_graph_2: nx.DiGraph,
                                  grid) -> Dict[FrozenSet[int], FrozenSet[int]]:
    """
    Match Morse sets between two graphs based on topological role and spatial proximity.

    Strategy:
    1. Classify nodes by topology: sources (in_degree=0), sinks (out_degree=0), intermediate
    2. Within each class, match by spatial proximity (nearest centroid)
    3. If class sizes differ, match as many as possible

    :param morse_graph_1: First Morse graph (test)
    :param morse_graph_2: Second Morse graph (ground truth)
    :param grid: Grid object for computing centroids
    :return: Dict mapping morse sets from graph_1 to graph_2
    """
    def classify_nodes(graph):
        """Classify nodes by topological role."""
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        intermediate = [n for n in graph.nodes() if graph.in_degree(n) > 0 and graph.out_degree(n) > 0]
        return sources, sinks, intermediate

    def match_by_proximity(nodes_1, nodes_2):
        """Match nodes within a class by spatial proximity."""
        if not nodes_1 or not nodes_2:
            return {}

        # Compute centroids
        centroids_1 = {n: compute_morse_set_centroid(grid, n) for n in nodes_1}
        centroids_2 = {n: compute_morse_set_centroid(grid, n) for n in nodes_2}

        # Greedy matching: for each node in nodes_1, find nearest in nodes_2
        matching = {}
        used = set()

        for n1 in nodes_1:
            c1 = centroids_1[n1]
            best_match = None
            best_dist = float('inf')

            for n2 in nodes_2:
                if n2 in used:
                    continue
                c2 = centroids_2[n2]
                dist = np.linalg.norm(c1 - c2)

                if dist < best_dist:
                    best_dist = dist
                    best_match = n2

            if best_match is not None:
                matching[n1] = best_match
                used.add(best_match)

        return matching

    # Classify nodes in both graphs
    sources_1, sinks_1, inter_1 = classify_nodes(morse_graph_1)
    sources_2, sinks_2, inter_2 = classify_nodes(morse_graph_2)

    # Match within each class
    matching = {}
    matching.update(match_by_proximity(sources_1, sources_2))
    matching.update(match_by_proximity(sinks_1, sinks_2))
    matching.update(match_by_proximity(inter_1, inter_2))

    return matching


def compute_basin_iou(basin_1: Set[int], basin_2: Set[int]) -> float:
    """
    Compute Intersection over Union (IoU) for two basins.

    IoU = |intersection| / |union|

    :param basin_1: Set of box indices in first basin
    :param basin_2: Set of box indices in second basin
    :return: IoU value in [0, 1]
    """
    intersection = basin_1 & basin_2
    union = basin_1 | basin_2

    if len(union) == 0:
        return 1.0  # Both empty → perfect match

    return len(intersection) / len(union)


def compute_roa_accuracy(morse_graph_test: nx.DiGraph,
                         basins_test: Dict[FrozenSet[int], Set[int]],
                         morse_graph_gt: nx.DiGraph,
                         basins_gt: Dict[FrozenSet[int], Set[int]],
                         grid) -> Dict:
    """
    Compute Region of Attraction (RoA) accuracy via IoU.

    Matches Morse sets topologically, then computes per-attractor IoU.

    :param morse_graph_test: Test Morse graph
    :param basins_test: Test basins
    :param morse_graph_gt: Ground truth Morse graph
    :param basins_gt: Ground truth basins
    :param grid: Grid object
    :return: Dict with 'per_attractor_iou' and 'mean_iou'
    """
    # Match morse sets
    matching = match_morse_sets_by_topology(morse_graph_test, morse_graph_gt, grid)

    # Compute IoU for each matched pair
    per_attractor_iou = {}

    for morse_set_test, morse_set_gt in matching.items():
        basin_test = basins_test.get(morse_set_test, set())
        basin_gt = basins_gt.get(morse_set_gt, set())

        iou = compute_basin_iou(basin_test, basin_gt)
        per_attractor_iou[morse_set_test] = iou

    # Compute mean IoU
    if per_attractor_iou:
        mean_iou = np.mean(list(per_attractor_iou.values()))
    else:
        mean_iou = 0.0

    return {
        'per_attractor_iou': per_attractor_iou,
        'mean_iou': mean_iou,
        'num_matched': len(matching)
    }


def compute_graph_metrics(morse_graph_1: nx.DiGraph,
                          morse_graph_2: nx.DiGraph) -> Dict:
    """
    Compute graph structure comparison metrics.

    :param morse_graph_1: First Morse graph
    :param morse_graph_2: Second Morse graph
    :return: Dict with isomorphism, edit distance, node/edge counts
    """
    num_nodes_1 = morse_graph_1.number_of_nodes()
    num_nodes_2 = morse_graph_2.number_of_nodes()
    num_edges_1 = morse_graph_1.number_of_edges()
    num_edges_2 = morse_graph_2.number_of_edges()

    # Check isomorphism
    try:
        isomorphic = nx.is_isomorphic(morse_graph_1, morse_graph_2)
    except:
        isomorphic = False

    # Compute graph edit distance (with timeout for large graphs)
    try:
        # Create simplified graphs with integer node labels for edit distance
        G1_simple = nx.DiGraph()
        G1_simple.add_nodes_from(range(num_nodes_1))
        node_map_1 = {node: i for i, node in enumerate(morse_graph_1.nodes())}
        G1_simple.add_edges_from([(node_map_1[u], node_map_1[v])
                                   for u, v in morse_graph_1.edges()])

        G2_simple = nx.DiGraph()
        G2_simple.add_nodes_from(range(num_nodes_2))
        node_map_2 = {node: i for i, node in enumerate(morse_graph_2.nodes())}
        G2_simple.add_edges_from([(node_map_2[u], node_map_2[v])
                                   for u, v in morse_graph_2.edges()])

        # Compute edit distance with timeout
        edit_distance = nx.graph_edit_distance(G1_simple, G2_simple, timeout=10)

        # Normalize by max number of nodes for interpretability
        if max(num_nodes_1, num_nodes_2) > 0:
            normalized_edit_distance = edit_distance / max(num_nodes_1, num_nodes_2)
        else:
            normalized_edit_distance = 0.0
    except:
        edit_distance = None
        normalized_edit_distance = None

    return {
        'isomorphic': isomorphic,
        'edit_distance': edit_distance,
        'normalized_edit_distance': normalized_edit_distance,
        'num_nodes': (num_nodes_1, num_nodes_2),
        'num_edges': (num_edges_1, num_edges_2),
        'node_diff': abs(num_nodes_1 - num_nodes_2),
        'edge_diff': abs(num_edges_1 - num_edges_2)
    }


def compute_betti_numbers(morse_graph: nx.DiGraph) -> List[int]:
    """
    Compute Betti numbers for a Morse graph.

    For directed acyclic graphs (DAGs):
    - β₀ = number of weakly connected components
    - β₁ = 0 (no cycles by definition)

    :param morse_graph: Morse graph (should be a DAG)
    :return: List of Betti numbers [β₀, β₁]
    """
    # β₀ = number of weakly connected components
    beta_0 = nx.number_weakly_connected_components(morse_graph)

    # β₁ = number of independent cycles
    # For DAGs, this should be 0
    # We can check by converting to undirected and looking for cycles
    undirected = morse_graph.to_undirected()
    try:
        # For a tree, β₁ = |E| - |V| + |C| where C = connected components
        num_edges = undirected.number_of_edges()
        num_nodes = undirected.number_of_nodes()
        num_components = nx.number_connected_components(undirected)

        beta_1 = num_edges - num_nodes + num_components
        beta_1 = max(0, beta_1)  # Should be 0 for DAGs
    except:
        beta_1 = 0

    return [beta_0, beta_1]


def compare_homology(morse_graph_1: nx.DiGraph,
                     morse_graph_2: nx.DiGraph) -> Dict:
    """
    Compare homology (Betti numbers) of two Morse graphs.

    :param morse_graph_1: First Morse graph
    :param morse_graph_2: Second Morse graph
    :return: Dict with Betti numbers and match status
    """
    betti_1 = compute_betti_numbers(morse_graph_1)
    betti_2 = compute_betti_numbers(morse_graph_2)

    betti_match = (betti_1 == betti_2)

    return {
        'betti_1': betti_1,
        'betti_2': betti_2,
        'betti_match': betti_match
    }


def compute_all_metrics(morse_graph_test: nx.DiGraph,
                        basins_test: Dict[FrozenSet[int], Set[int]],
                        morse_graph_gt: nx.DiGraph,
                        basins_gt: Dict[FrozenSet[int], Set[int]],
                        grid) -> Dict:
    """
    Compute all metrics comparing test against ground truth.

    :param morse_graph_test: Test Morse graph
    :param basins_test: Test basins
    :param morse_graph_gt: Ground truth Morse graph
    :param basins_gt: Ground truth basins
    :param grid: Grid object
    :return: Comprehensive dict with all metrics
    """
    roa_metrics = compute_roa_accuracy(morse_graph_test, basins_test,
                                       morse_graph_gt, basins_gt, grid)

    graph_metrics = compute_graph_metrics(morse_graph_test, morse_graph_gt)

    return {
        'roa': roa_metrics,
        'graph': graph_metrics
    }
