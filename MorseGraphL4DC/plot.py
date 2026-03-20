import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Dict, Set, FrozenSet

from .grids import AbstractGrid

def plot_morse_sets(grid: AbstractGrid, morse_graph: nx.DiGraph, ax: plt.Axes = None,
                   box_map: nx.DiGraph = None, show_outside: bool = False,
                   dims=(0, 1), **kwargs):
    """
    Plots the Morse sets on a grid.

    :param grid: The grid object used for the computation.
    :param morse_graph: The Morse graph (NetworkX DiGraph) containing the Morse sets as nodes.
                       Each node should have a 'color' attribute (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param box_map: The BoxMap (directed graph) from compute_box_map(). Required if show_outside=True.
    :param show_outside: If True, paint boxes that map outside the domain in grey. Requires box_map.
    :param kwargs: Additional keyword arguments to pass to the PatchCollection.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Extract morse sets from the NetworkX graph
    morse_sets = morse_graph.nodes()

    # Get all boxes from the grid
    all_boxes = grid.get_boxes()

    rects = []
    colors = []

    # Use colors from node attributes (assigned by compute_morse_graph)
    for morse_set in morse_sets:
        # Get color from node attribute, fallback to viridis if not present
        if 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Fallback for backward compatibility (colorblind-friendly)
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('viridis')
            color = cmap(list(morse_sets).index(morse_set) / max(num_sets, 10))

        for box_index in morse_set:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                d0, d1 = dims
                rect = Rectangle((box[0, d0], box[0, d1]),
                               box[1, d0] - box[0, d0],
                               box[1, d1] - box[0, d1])
                rects.append(rect)
                colors.append(color)

    if rects:
        pc = PatchCollection(rects, facecolors=colors, alpha=0.7, **kwargs)
        ax.add_collection(pc)

    # Optionally show boxes that map outside the domain (have data but no transitions)
    if show_outside and box_map is not None:
        # Collect all boxes in Morse sets
        morse_set_boxes = set()
        for morse_set in morse_sets:
            morse_set_boxes.update(morse_set)

        # Find boxes with out_degree == 0 that are not in any Morse set
        # These boxes have data but map outside the domain
        outside_boxes = set()
        for node in box_map.nodes():
            if box_map.out_degree(node) == 0 and node not in morse_set_boxes:
                outside_boxes.add(node)

        # Paint outside boxes grey
        d0, d1 = dims
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, d0], box[0, d1]),
                               box[1, d0] - box[0, d0],
                               box[1, d1] - box[0, d1],
                               facecolor='grey',
                               edgecolor='none',
                               alpha=0.4)
                ax.add_patch(rect)

    d0, d1 = dims
    ax.set_xlim(grid.bounds[0, d0], grid.bounds[1, d0])
    ax.set_ylim(grid.bounds[0, d1], grid.bounds[1, d1])
    ax.set_aspect('equal', adjustable='box')

def plot_basins_of_attraction(grid: AbstractGrid, basins,
                             morse_graph: nx.DiGraph = None, ax: plt.Axes = None,
                             show_outside: bool = False, dims=(0, 1), **kwargs):
    """
    Plots the basins of attraction with colors matching the Morse sets.

    :param grid: The grid object used for the computation.
    :param basins: Dict[FrozenSet[int], Set[int]]: box-level basins (from compute_all_morse_set_basins)
    :param morse_graph: The Morse graph with color attributes. If provided, uses colors from node attributes.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param show_outside: If True, paint boxes not in any basin black (boxes mapping outside domain).
    :param kwargs: Additional keyword arguments to pass to Rectangle patches.
    """
    if ax is None:
        _, ax = plt.subplots()

    all_boxes = grid.get_boxes()

    # Plot each basin
    for attractor, basin in basins.items():
        # Get color from morse_graph node attributes if available
        if morse_graph and 'color' in morse_graph.nodes[attractor]:
            color = morse_graph.nodes[attractor]['color']
        else:
            # Fallback to viridis colormap for backward compatibility
            colors_cmap = plt.cm.get_cmap('viridis', len(basins))
            color = colors_cmap(list(basins.keys()).index(attractor))

        d0, d1 = dims
        # Box-level basins: basin is a set of box indices
        # Plot basin boxes with lower opacity
        for box_index in basin:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, d0], box[0, d1]),
                               box[1, d0] - box[0, d0],
                               box[1, d1] - box[0, d1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=0.3, **kwargs)
                ax.add_patch(rect)

        # Plot attractor itself with full opacity
        for box_index in attractor:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, d0], box[0, d1]),
                               box[1, d0] - box[0, d0],
                               box[1, d1] - box[0, d1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=1.0, **kwargs)
                ax.add_patch(rect)

    # Optionally show boxes that don't belong to any basin (map outside domain)
    if show_outside:
        all_box_indices = set(range(len(all_boxes)))
        # For box basins, union all basin box sets
        basin_box_indices = set().union(*basins.values())
        outside_boxes = all_box_indices - basin_box_indices

        d0, d1 = dims
        # Paint outside boxes black
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, d0], box[0, d1]),
                               box[1, d0] - box[0, d0],
                               box[1, d1] - box[0, d1],
                               facecolor='black',
                               edgecolor='none',
                               alpha=0.5)
                ax.add_patch(rect)

    d0, d1 = dims
    ax.set_xlim(grid.bounds[0, d0], grid.bounds[1, d0])
    ax.set_ylim(grid.bounds[0, d1], grid.bounds[1, d1])
    ax.set_aspect('equal', adjustable='box')

def _hierarchical_dag_layout(G):
    """
    Compute a hierarchical layout for a directed acyclic graph (DAG).

    Nodes are arranged in layers based on their longest path to a source node.
    Nodes with no incoming edges are placed at the top; sinks at the bottom.

    :param G: A networkx DiGraph (should be acyclic)
    :return: A dictionary mapping nodes to (x, y) positions
    """
    # Compute longest path from each node to any source (node with in-degree 0)
    # This determines the "level" or "layer" of each node

    # For DAGs, we can use a topological sort and compute levels
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        # Not a DAG, return empty to trigger fallback
        raise ValueError("Graph is not a DAG")

    # Compute the level of each node (distance from sources)
    levels = {}
    for node in topo_order:
        # Find maximum level of predecessors
        predecessors = list(G.predecessors(node))
        if not predecessors:
            # Source node
            levels[node] = 0
        else:
            # Level is max level of predecessors + 1
            levels[node] = max(levels[pred] for pred in predecessors) + 1

    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Create positions: y-coordinate is level (inverted so 0 is at top)
    # x-coordinate spreads nodes horizontally within each level
    pos = {}
    max_level = max(levels.values()) if levels else 0

    for level in sorted(level_groups.keys()):
        nodes_in_level = level_groups[level]
        num_nodes = len(nodes_in_level)

        # Spread nodes horizontally
        for i, node in enumerate(nodes_in_level):
            if num_nodes == 1:
                x = 0
            else:
                x = (i - (num_nodes - 1) / 2) * 2

            # y goes from top (max_level) to bottom (0)
            y = max_level - level
            pos[node] = (x, y)

    return pos

def plot_morse_graph(morse_graph: nx.DiGraph, ax: plt.Axes = None,
                    morse_sets_colors: dict = None, node_size: int = 300,
                    arrowsize: int = 20, font_size: int = 8, show_box_counts: bool = False,
                    show_title: bool = False):
    """
    Plots the Morse graph with hierarchical layout.

    The graph is displayed as a directed acyclic graph (DAG) where nodes at the top
    represent sources (recurrent sets with no incoming edges from other recurrent sets)
    and nodes at the bottom represent sinks. This visualizes the partial order on
    recurrent components.

    :param morse_graph: The Morse graph to plot. Each node should have a 'color' attribute
                       (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param morse_sets_colors: Deprecated parameter, ignored. Colors are taken from node attributes.
    :param node_size: Size of the nodes.
    :param arrowsize: Size of the arrow heads.
    :param font_size: Font size for node labels.
    :param show_box_counts: If True, display the number of boxes in each Morse set next to the node number.
    :param show_title: If True, display the title on the axes. Default False for paper-ready figures.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())

    # Use colors from node attributes or generate
    node_colors = []
    for morse_set in morse_sets:
        color = None
        if 'color' in morse_graph.nodes[morse_set]:
            # Use color from node attribute
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Generate color for backward compatibility (colorblind-friendly)
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('viridis')
            color = cmap(morse_sets.index(morse_set) / max(num_sets, 10))

        # Convert numpy floats to python floats to avoid pygraphviz warning
        if hasattr(color, '__iter__'):
            color = tuple(float(c) for c in color)

        node_colors.append(color)
        # Update graph attribute for pygraphviz
        morse_graph.nodes[morse_set]['color'] = color

    # Create a mapping from frozenset to a shorter string representation
    if show_box_counts:
        node_labels = {node: f"{i+1}\n({len(node)})"
                      for i, node in enumerate(morse_sets)}
    else:
        node_labels = {node: str(i+1) for i, node in enumerate(morse_sets)}

    # Try hierarchical layout, with smart DAG-aware fallback
    try:
        from networkx.drawing.nx_agraph import pygraphviz_layout
        pos = pygraphviz_layout(morse_graph, prog='dot')
    except (ImportError, Exception):
        # Fallback: use a topological sort-based hierarchical layout
        try:
            pos = _hierarchical_dag_layout(morse_graph)
        except:
            # Final fallback: spring layout
            pos = nx.spring_layout(morse_graph, seed=42)

    # Center the layout by shifting positions so centroid is at origin
    if pos:
        pos_array = np.array(list(pos.values()))
        centroid = pos_array.mean(axis=0)
        pos = {node: (x - centroid[0], y - centroid[1]) for node, (x, y) in pos.items()}
        
        # Calculate symmetric axis limits to ensure centered display
        pos_array_centered = np.array(list(pos.values()))
        if len(pos_array_centered) > 0:
            max_x = np.abs(pos_array_centered[:, 0]).max()
            max_y = np.abs(pos_array_centered[:, 1]).max()
            # Add some padding (10% margin)
            padding = 0.1
            xlim = (-max_x * (1 + padding), max_x * (1 + padding))
            ylim = (-max_y * (1 + padding), max_y * (1 + padding))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Draw the graph components
    # Note: node_colors are RGBA tuples which matplotlib handles correctly
    nx.draw_networkx_nodes(morse_graph, pos, node_color=node_colors,
                          node_size=node_size, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(morse_graph, pos, edge_color='gray',
                          arrows=True, arrowsize=arrowsize, ax=ax, alpha=0.6,
                          connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(morse_graph, pos, labels=node_labels,
                           font_size=font_size, ax=ax)

    if show_title:
        ax.set_title("Morse Graph (DAG of Recurrent Components)")
    ax.axis('off')

def plot_morse_graph_academic(morse_graph: nx.DiGraph, ax: plt.Axes = None,
                             morse_sets_colors: dict = None, 
                             node_size: int = 800,  # Larger nodes for visibility
                             edge_width: float = 2.0,  # Thicker edges
                             edge_alpha: float = 0.9,  # More opaque edges
                             edge_color: str = 'black',  # Black instead of gray
                             arrowsize: int = 30,  # Larger arrows
                             font_size: int = 12,  # Larger font
                             font_weight: str = 'bold',  # Bold text
                             show_box_counts: bool = False,
                             show_title: bool = False,
                             node_edge_width: float = 1.5,  # Add border to nodes
                             node_edge_color: str = 'black'):  # Black border for definition
    """
    Plots the Morse graph optimized for academic papers with small text widths.
    
    This version uses larger, more prominent visual elements suitable for
    publication in academic papers where figures may be displayed at small sizes.
    
    :param morse_graph: The Morse graph to plot. Each node should have a 'color' attribute
                       (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param morse_sets_colors: Deprecated parameter, ignored. Colors are taken from node attributes.
    :param node_size: Size of the nodes (default 800 for better visibility).
    :param edge_width: Width of edges (default 2.0 for better visibility).
    :param edge_alpha: Alpha value for edges (default 0.9, nearly opaque).
    :param edge_color: Color of edges (default 'black' for better contrast).
    :param arrowsize: Size of the arrow heads (default 30).
    :param font_size: Font size for node labels (default 12).
    :param font_weight: Font weight ('bold' by default for better readability).
    :param show_box_counts: If True, display the number of boxes in each Morse set.
    :param show_title: If True, display the title on the axes. Default False for papers.
    :param node_edge_width: Width of the border around nodes.
    :param node_edge_color: Color of the border around nodes.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())

    # Use colors from node attributes or generate
    node_colors = []
    for morse_set in morse_sets:
        color = None
        if 'color' in morse_graph.nodes[morse_set]:
            # Use color from node attribute
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Generate color for backward compatibility (high contrast)
            num_sets = len(morse_sets)
            # Use tab10 for better contrast in print
            cmap = cm.get_cmap('tab10' if num_sets <= 10 else 'viridis')
            color = cmap(morse_sets.index(morse_set) / max(num_sets - 1, 1))

        # Convert numpy floats to python floats
        if hasattr(color, '__iter__'):
            color = tuple(float(c) for c in color)

        node_colors.append(color)
        # Update graph attribute
        morse_graph.nodes[morse_set]['color'] = color

    # Create node labels
    if show_box_counts:
        node_labels = {node: f"{i+1}\n({len(node)})"
                      for i, node in enumerate(morse_sets)}
    else:
        node_labels = {node: str(i+1) for i, node in enumerate(morse_sets)}

    # Try hierarchical layout
    try:
        from networkx.drawing.nx_agraph import pygraphviz_layout
        pos = pygraphviz_layout(morse_graph, prog='dot')
    except (ImportError, Exception):
        # Fallback: use a topological sort-based hierarchical layout
        try:
            pos = _hierarchical_dag_layout(morse_graph)
        except:
            # Final fallback: spring layout
            pos = nx.spring_layout(morse_graph, seed=42)

    # Center the layout
    if pos:
        pos_array = np.array(list(pos.values()))
        centroid = pos_array.mean(axis=0)
        pos = {node: (x - centroid[0], y - centroid[1]) for node, (x, y) in pos.items()}
        
        # Calculate symmetric axis limits
        pos_array_centered = np.array(list(pos.values()))
        if len(pos_array_centered) > 0:
            max_x = np.abs(pos_array_centered[:, 0]).max()
            max_y = np.abs(pos_array_centered[:, 1]).max()
            # Add some padding (15% margin for academic figures)
            padding = 0.15
            xlim = (-max_x * (1 + padding), max_x * (1 + padding))
            ylim = (-max_y * (1 + padding), max_y * (1 + padding))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(morse_graph, pos, 
                          edge_color=edge_color,
                          width=edge_width,
                          arrows=True, 
                          arrowsize=arrowsize, 
                          ax=ax, 
                          alpha=edge_alpha,
                          connectionstyle="arc3,rad=0.1",
                          arrowstyle='-|>',  # Better arrow style
                          min_source_margin=15,  # Account for larger nodes
                          min_target_margin=15)
    
    # Draw nodes with border for better definition
    nx.draw_networkx_nodes(morse_graph, pos, 
                          node_color=node_colors,
                          node_size=node_size, 
                          ax=ax, 
                          alpha=1.0,  # Full opacity
                          linewidths=node_edge_width,
                          edgecolors=node_edge_color)
    
    # Draw labels with better visibility
    nx.draw_networkx_labels(morse_graph, pos, 
                           labels=node_labels,
                           font_size=font_size, 
                           font_weight=font_weight,
                           font_color='white' if edge_color == 'black' else 'black',
                           ax=ax)

    if show_title:
        ax.set_title("Morse Graph", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Make the plot tighter for academic papers
    ax.set_aspect('equal')

def visualize_morse_sets_graph_basins(grid, morse_graph, basins, box_map,
                                       method_label, bounds, labels, output_path,
                                       show_outside=False):
    """
    Create 3-panel visualization: Morse Sets | Morse Graph | Basins of Attraction.

    This is a pure visualization function that creates the standard 3-panel figure
    for Morse graph analysis. Separated from computation for modularity.
    Now uses academic style for the Morse graph panel for better visibility.

    :param grid: UniformGrid object
    :param morse_graph: NetworkX DiGraph of morse sets
    :param basins: Dict mapping morse sets (frozensets) to their basins (sets of box indices)
    :param box_map: NetworkX DiGraph of box-to-box transitions (for outside detection)
    :param method_label: Display label for plots (e.g., 'F_integration (f)')
    :param bounds: Domain bounds array of shape (2, D) [[mins], [maxs]]
    :param labels: Tuple of (xlabel, ylabel) for axis labels
    :param output_path: Full path to save the figure (e.g., './toggle_switch_integration_f.png')
    :param show_outside: Whether to highlight boxes that map outside (for F_data)
    :return: output_path

    Example:
        >>> from MorseGraphL4DC.analysis import compute_morse_graph_analysis
        >>> from MorseGraphL4DC.plot import visualize_morse_sets_graph_basins
        >>> box_map, morse_graph, basins = compute_morse_graph_analysis(grid, F)
        >>> visualize_morse_sets_graph_basins(
        ...     grid, morse_graph, basins, box_map,
        ...     'F_integration', bounds, ('x1', 'x2'), './output.png'
        ... )
    """
    import matplotlib.pyplot as plt
    import os

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    xlabel, ylabel = labels
    lower_bounds, upper_bounds = bounds[0], bounds[1]

    # Panel 1: Morse Sets
    plot_morse_sets(grid, morse_graph, ax=axes[0],
                    box_map=box_map if show_outside else None,
                    show_outside=show_outside)
    axes[0].set_xlim(lower_bounds[0], upper_bounds[0])
    axes[0].set_ylim(lower_bounds[1], upper_bounds[1])
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # Panel 2: Morse Graph - Use academic style for better visibility
    plot_morse_graph_academic(morse_graph, ax=axes[1])

    # Panel 3: Basins of Attraction
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=axes[2],
                              show_outside=show_outside)
    axes[2].set_xlim(lower_bounds[0], upper_bounds[0])
    axes[2].set_ylim(lower_bounds[1], upper_bounds[1])
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def save_morse_sets_only(grid, morse_graph, box_map, output_path, bounds, show_outside=False):
    """
    Save only the Morse sets panel as a standalone figure (paper-ready, no title).

    :param grid: UniformGrid object
    :param morse_graph: NetworkX DiGraph of morse sets
    :param box_map: NetworkX DiGraph of box-to-box transitions
    :param output_path: Full path to save the figure
    :param bounds: Domain bounds array of shape (2, D) [[mins], [maxs]]
    :param show_outside: Whether to highlight boxes that map outside
    :return: output_path
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    lower_bounds, upper_bounds = bounds[0], bounds[1]

    plot_morse_sets(grid, morse_graph, ax=ax,
                   box_map=box_map if show_outside else None,
                   show_outside=show_outside)
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # No title for paper-ready figure

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def save_morse_graph_only(morse_graph, output_path, show_box_counts=False):
    """
    Save only the Morse graph (DAG) panel as a standalone figure (paper-ready, no title).
    Uses academic style for better visibility in papers.

    :param morse_graph: NetworkX DiGraph of morse sets
    :param output_path: Full path to save the figure
    :param show_box_counts: Whether to show box counts in node labels
    :return: output_path
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Use academic style for paper-ready figures
    plot_morse_graph_academic(morse_graph, ax=ax, show_box_counts=show_box_counts)
    # No title for paper-ready figure

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def save_basins_only(grid, basins, morse_graph, output_path, bounds, show_outside=False):
    """
    Save only the basins of attraction panel as a standalone figure (paper-ready, no title).

    :param grid: UniformGrid object
    :param basins: Dict mapping morse sets (frozensets) to their basins (sets of box indices)
    :param morse_graph: NetworkX DiGraph of morse sets
    :param output_path: Full path to save the figure
    :param bounds: Domain bounds array of shape (2, D) [[mins], [maxs]]
    :param show_outside: Whether to show boxes outside any basin
    :return: output_path
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    lower_bounds, upper_bounds = bounds[0], bounds[1]

    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=ax,
                             show_outside=show_outside)
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # No title for paper-ready figure

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path

def save_all_panels_individually(grid, morse_graph, basins, box_map, base_name, bounds,
                                  method_label='', labels=('$x_1$', '$x_2$'), show_outside=False):
    """
    Save all visualizations: individual panels (paper-ready) + combined 3-panel figure.

    Creates 4 output files:
    - {base_name}_morse_sets.png (paper-ready, no title)
    - {base_name}_morse_graph.png (paper-ready, no title)
    - {base_name}_basins.png (paper-ready, no title)
    - {base_name}.png (combined 3-panel, no titles)

    :param grid: UniformGrid object
    :param morse_graph: NetworkX DiGraph of morse sets
    :param basins: Dict mapping morse sets to their basins
    :param box_map: NetworkX DiGraph of box-to-box transitions
    :param base_name: Base path without extension (e.g., 'output/method1_integration_f')
    :param bounds: Domain bounds array of shape (2, D)
    :param method_label: Label for combined figure titles (e.g., 'F_integration (f)')
    :param labels: Tuple of (xlabel, ylabel) for axis labels
    :param show_outside: Whether to highlight boxes that map outside
    :return: Dict with keys 'morse_sets', 'morse_graph', 'basins', 'combined' containing paths
    """
    paths = {}

    # Save individual panels (paper-ready, no titles)
    paths['morse_sets'] = save_morse_sets_only(
        grid, morse_graph, box_map,
        f"{base_name}_morse_sets.png",
        bounds, show_outside
    )

    paths['morse_graph'] = save_morse_graph_only(
        morse_graph,
        f"{base_name}_morse_graph.png"
    )

    paths['basins'] = save_basins_only(
        grid, basins, morse_graph,
        f"{base_name}_basins.png",
        bounds, show_outside
    )

    # Save combined 3-panel figure (with titles, for overview)
    paths['combined'] = visualize_morse_sets_graph_basins(
        grid, morse_graph, basins, box_map,
        method_label,
        bounds, labels,
        f"{base_name}.png",
        show_outside
    )

    return paths
