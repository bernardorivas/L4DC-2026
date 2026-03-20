"""
Comparison utilities for Morse graph methods.

Provides metrics for comparing different F(rect) computation methods,
including IoU (Intersection over Union) for Morse sets and basins.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def compute_morse_set_iou_table(combined_morse_graphs: List,
                                  method_names: List[str],
                                  ground_truth_idx: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Compute IoU table for Morse sets comparing methods against ground truth.

    Supports flexible node counts (2 nodes, 3 nodes, etc.)
    If a method has fewer nodes than ground truth, missing nodes are treated as empty (IoU = 0)

    :param combined_morse_graphs: List of collapsed Morse graphs (one per method)
    :param method_names: List of method names corresponding to graphs
    :param ground_truth_idx: Index of ground truth method (default: 0 = F_integration(f))
    :return: Dict with structure {method_name: {node_0: iou, node_1: iou, ..., mean: iou}}

    Example:
        >>> graphs = [morse_graph1, morse_graph2, morse_graph3]
        >>> methods = ['F_integration(f)', 'F_data', 'F_Lipschitz']
        >>> iou_table = compute_morse_set_iou_table(graphs, methods, ground_truth_idx=0)
        >>> # iou_table['F_data']['node_0'] gives IoU for node 0
    """
    gt_graph = combined_morse_graphs[ground_truth_idx]
    gt_nodes = list(gt_graph.nodes())
    num_nodes = len(gt_nodes)

    iou_table = {}

    for i, method_name in enumerate(method_names):
        if i == ground_truth_idx:
            # Skip ground truth (IoU = 1.0 with itself)
            continue

        method_graph = combined_morse_graphs[i]
        method_nodes = list(method_graph.nodes())
        num_method_nodes = len(method_nodes)

        # Handle node count mismatch
        if num_method_nodes > num_nodes:
            raise ValueError(
                f"Method {method_name} has MORE nodes ({num_method_nodes}) than "
                f"ground truth ({num_nodes}). This is unexpected."
            )

        ious = []
        iou_dict = {}

        for node_idx in range(num_nodes):
            if node_idx < num_method_nodes:
                # Node exists in method
                gt_boxes = set(gt_nodes[node_idx])
                method_boxes = set(method_nodes[node_idx])

                # Compute IoU
                intersection = gt_boxes & method_boxes
                union = gt_boxes | method_boxes

                if len(union) == 0:
                    iou = 0.0  # Both empty (shouldn't happen)
                else:
                    iou = len(intersection) / len(union)
            else:
                # Node missing in method (treat as empty set)
                # IoU = 0 since intersection is empty
                iou = 0.0

            ious.append(iou)
            iou_dict[f'node_{node_idx}'] = iou

        # Add mean IoU
        iou_dict['mean'] = np.mean(ious)
        iou_table[method_name] = iou_dict

    return iou_table


def compute_roa_iou_table(combined_roas_list: List[Dict],
                              combined_morse_graphs: List,
                              method_names: List[str],
                              ground_truth_idx: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Compute IoU table for RoAs (regions of attraction) comparing methods against ground truth.

    Supports flexible node counts (2 nodes, 3 nodes, etc.)
    If a method has fewer nodes than ground truth, missing nodes are treated as empty (IoU = 0)

    :param combined_roas_list: List of RoA dicts (one per method)
    :param combined_morse_graphs: List of collapsed Morse graphs (for node ordering)
    :param method_names: List of method names
    :param ground_truth_idx: Index of ground truth method (default: 0)
    :return: Dict with structure {method_name: {node_0: iou, node_1: iou, ..., mean: iou}}

    Note: RoAs include the Morse set boxes themselves (computed with containment algorithm)
    """
    gt_roas = combined_roas_list[ground_truth_idx]
    gt_graph = combined_morse_graphs[ground_truth_idx]
    gt_nodes = list(gt_graph.nodes())
    num_nodes = len(gt_nodes)

    iou_table = {}

    for i, method_name in enumerate(method_names):
        if i == ground_truth_idx:
            # Skip ground truth
            continue

        method_roas = combined_roas_list[i]
        method_graph = combined_morse_graphs[i]
        method_nodes = list(method_graph.nodes())
        num_method_nodes = len(method_nodes)

        # Handle node count mismatch
        if num_method_nodes > num_nodes:
            raise ValueError(
                f"Method {method_name} has MORE nodes ({num_method_nodes}) than "
                f"ground truth ({num_nodes}). This is unexpected."
            )

        ious = []
        iou_dict = {}

        for node_idx in range(num_nodes):
            if node_idx < num_method_nodes:
                # Node exists in method
                gt_morse_set = gt_nodes[node_idx]
                method_morse_set = method_nodes[node_idx]

                gt_roa_boxes = set(gt_roas[gt_morse_set])
                method_roa_boxes = set(method_roas[method_morse_set])

                # Compute IoU
                intersection = gt_roa_boxes & method_roa_boxes
                union = gt_roa_boxes | method_roa_boxes

                if len(union) == 0:
                    iou = 0.0
                else:
                    iou = len(intersection) / len(union)
            else:
                # Node missing in method (treat as empty RoA)
                # IoU = 0 since intersection is empty
                iou = 0.0

            ious.append(iou)
            iou_dict[f'node_{node_idx}'] = iou

        # Add mean IoU
        iou_dict['mean'] = np.mean(ious)
        iou_table[method_name] = iou_dict

    return iou_table


def compute_coverage_ratios(combined_morse_graphs: List,
                              method_names: List[str],
                              ground_truth_idx: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Compute coverage ratios (volume ratios) for Morse sets.

    Supports flexible node counts (2 nodes, 3 nodes, etc.)
    If a method has fewer nodes than ground truth, missing nodes have ratio = 0.0

    Ratio > 1 indicates over-approximation (conservative)
    Ratio < 1 indicates under-approximation (aggressive)
    Ratio = 0 indicates node not found (missing attractor/repeller)

    :param combined_morse_graphs: List of collapsed Morse graphs
    :param method_names: List of method names
    :param ground_truth_idx: Index of ground truth method
    :return: Dict with structure {method_name: {node_0: ratio, node_1: ratio, ...}}
    """
    gt_graph = combined_morse_graphs[ground_truth_idx]
    gt_nodes = list(gt_graph.nodes())
    num_nodes = len(gt_nodes)

    ratio_table = {}

    for i, method_name in enumerate(method_names):
        if i == ground_truth_idx:
            continue

        method_graph = combined_morse_graphs[i]
        method_nodes = list(method_graph.nodes())
        num_method_nodes = len(method_nodes)

        # Handle node count mismatch
        if num_method_nodes > num_nodes:
            raise ValueError(
                f"Method {method_name} has MORE nodes ({num_method_nodes}) than "
                f"ground truth ({num_nodes}). This is unexpected."
            )

        ratio_dict = {}

        for node_idx in range(num_nodes):
            if node_idx < num_method_nodes:
                # Node exists in method
                gt_size = len(gt_nodes[node_idx])
                method_size = len(method_nodes[node_idx])

                if gt_size == 0:
                    ratio = float('inf') if method_size > 0 else 1.0
                else:
                    ratio = method_size / gt_size
            else:
                # Node missing in method
                ratio = 0.0

            ratio_dict[f'node_{node_idx}'] = ratio

        ratio_table[method_name] = ratio_dict

    return ratio_table


def format_iou_tables_text(morse_set_iou: Dict,
                             roa_iou: Dict,
                             coverage_ratios: Dict,
                             ground_truth_name: str = 'F_integration(f)',
                             node_labels: list = None) -> str:
    """
    Format IoU tables as pretty text output.

    :param morse_set_iou: Morse set IoU table from compute_morse_set_iou_table
    :param roa_iou: RoA IoU table from compute_roa_iou_table
    :param coverage_ratios: Coverage ratios from compute_coverage_ratios
    :param ground_truth_name: Name of ground truth method
    :param node_labels: List of node labels (e.g., ['Node 0 (Attractor)', ...])
                       If None, infers from number of nodes in data
    :return: Formatted string with tables
    """
    lines = []

    # Extract method names (excluding ground truth)
    method_names = list(morse_set_iou.keys())

    # Infer number of nodes from first method's data
    if method_names:
        num_nodes = len([k for k in morse_set_iou[method_names[0]].keys() if k.startswith('node_')])
    else:
        num_nodes = 3  # Default

    # Generate default labels if not provided
    if node_labels is None:
        if num_nodes == 2:
            node_labels = ['Node 0 (Limit Cycle)', 'Node 1 (Unstable Origin)']
        elif num_nodes == 3:
            node_labels = ['Node 0 (Attractor)', 'Node 1 (Attractor)', 'Node 2 (Repeller)']
        else:
            node_labels = [f'Node {i}' for i in range(num_nodes)]

    # Header
    lines.append("=" * 80)
    lines.append(f"MORSE SET IoU vs {ground_truth_name}")
    lines.append("=" * 80)

    # Column headers
    header = f"{'':20}"
    for method in method_names:
        header += f"{method:20}"
    lines.append(header)
    lines.append("-" * 80)

    # Node rows
    for node_idx in range(num_nodes):
        label = node_labels[node_idx] if node_idx < len(node_labels) else f'Node {node_idx}'
        row = f"{label:20}"
        for method in method_names:
            iou = morse_set_iou[method][f'node_{node_idx}']
            row += f"{iou:20.3f}"
        lines.append(row)

    # RoA IoU table
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"RoA IoU vs {ground_truth_name} [including Morse sets]")
    lines.append("=" * 80)

    # Column headers
    header = f"{'':20}"
    for method in method_names:
        header += f"{method:20}"
    lines.append(header)
    lines.append("-" * 80)

    # Node rows - use RoA suffix
    roa_labels = [f'{label.split(" (")[0]} RoA' for label in node_labels]
    for node_idx in range(num_nodes):
        label = roa_labels[node_idx] if node_idx < len(roa_labels) else f'Node {node_idx} RoA'
        row = f"{label:20}"
        for method in method_names:
            iou = roa_iou[method][f'node_{node_idx}']
            row += f"{iou:20.3f}"
        lines.append(row)

    # Coverage ratios table
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"SPATIAL COVERAGE (Volume Ratios vs {ground_truth_name})")
    lines.append("=" * 80)
    lines.append("Ratio > 1.0 indicates over-approximation (conservative)")
    lines.append("Ratio < 1.0 indicates under-approximation (aggressive)")
    lines.append("-" * 80)

    # Column headers
    header = f"{'':20}"
    for method in method_names:
        header += f"{method:20}"
    lines.append(header)
    lines.append("-" * 80)

    # Node rows - simple node names for coverage
    simple_labels = [f'Node {i}' for i in range(num_nodes)]
    for node_idx in range(num_nodes):
        label = simple_labels[node_idx]
        row = f"{label:20}"
        for method in method_names:
            ratio = coverage_ratios[method][f'node_{node_idx}']
            row += f"{ratio:20.3f}"
        lines.append(row)

    return '\n'.join(lines)
