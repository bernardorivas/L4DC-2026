"""
Caching utilities for saving and loading expensive Morse graph computations.

Provides functions to cache:
- Box maps (NetworkX DiGraph)
- Morse graphs (NetworkX DiGraph with frozenset nodes)
- Basins/RoAs (Dict[FrozenSet[int], Set[int]])
- Combined/post-processed Morse graphs and basins
- Shared data (trajectories, GP models)
"""

import os
import pickle
import json
import numpy as np
import networkx as nx
from typing import Dict, Set, FrozenSet, Optional, Any


def save_method_results(output_dir: str, method_num: int,
                        box_map: nx.DiGraph,
                        morse_graph: nx.DiGraph,
                        basins: Dict[FrozenSet[int], Set[int]],
                        combined_morse_graph: nx.DiGraph,
                        combined_roas: Dict[FrozenSet[int], Set[int]],
                        metadata: Dict[str, Any]) -> None:
    """
    Save computation results for a specific method.

    Creates directory structure:
        {output_dir}/computation/method{N}/
            box_map.pkl
            morse_graph.pkl
            basins.pkl
            combined_morse_graph.pkl
            combined_roas.pkl
            metadata.json

    :param output_dir: Output directory (e.g., 'example_1_tau_1_0')
    :param method_num: Method number (1-5)
    :param box_map: Box map graph
    :param morse_graph: Morse graph
    :param basins: Basins for each Morse set (reachability algorithm)
    :param combined_morse_graph: Post-processed Morse graph
    :param combined_roas: RoAs for combined Morse sets (containment algorithm)
    :param metadata: Dict with computation_time, etc.
    """
    method_dir = os.path.join(output_dir, 'computation', f'method{method_num}')
    os.makedirs(method_dir, exist_ok=True)

    # Save graphs with pickle (NetworkX graphs are picklable)
    with open(os.path.join(method_dir, 'box_map.pkl'), 'wb') as f:
        pickle.dump(box_map, f)
    with open(os.path.join(method_dir, 'morse_graph.pkl'), 'wb') as f:
        pickle.dump(morse_graph, f)
    with open(os.path.join(method_dir, 'combined_morse_graph.pkl'), 'wb') as f:
        pickle.dump(combined_morse_graph, f)

    # Save basins and RoAs with pickle (handles frozensets and sets)
    with open(os.path.join(method_dir, 'basins.pkl'), 'wb') as f:
        pickle.dump(basins, f)
    with open(os.path.join(method_dir, 'combined_roas.pkl'), 'wb') as f:
        pickle.dump(combined_roas, f)

    # Save metadata as JSON (human-readable)
    with open(os.path.join(method_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def load_method_results(output_dir: str, method_num: int) -> Optional[Dict[str, Any]]:
    """
    Load computation results for a specific method.

    :param output_dir: Output directory (e.g., 'example_1_tau_1_0')
    :param method_num: Method number (1-5)
    :return: Dict with keys: box_map, morse_graph, basins, combined_morse_graph, combined_roas, metadata
             Returns None if cache doesn't exist
    """
    method_dir = os.path.join(output_dir, 'computation', f'method{method_num}')

    # Check if all required files exist
    required_files = ['box_map.pkl', 'morse_graph.pkl', 'combined_morse_graph.pkl', 'metadata.json']
    for fname in required_files:
        if not os.path.exists(os.path.join(method_dir, fname)):
            return None

    # Load all files
    try:
        with open(os.path.join(method_dir, 'box_map.pkl'), 'rb') as f:
            box_map = pickle.load(f)
        with open(os.path.join(method_dir, 'morse_graph.pkl'), 'rb') as f:
            morse_graph = pickle.load(f)
        with open(os.path.join(method_dir, 'combined_morse_graph.pkl'), 'rb') as f:
            combined_morse_graph = pickle.load(f)
        with open(os.path.join(method_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # Load basins and roas if they exist (for backwards compatibility)
        basins = {}
        combined_roas = {}
        
        basins_path = os.path.join(method_dir, 'basins.pkl')
        if os.path.exists(basins_path):
            with open(basins_path, 'rb') as f:
                basins = pickle.load(f)
        
        # Try new name first, then old name for backwards compatibility
        roas_path = os.path.join(method_dir, 'combined_roas.pkl')
        old_roas_path = os.path.join(method_dir, 'combined_basins.pkl')
        if os.path.exists(roas_path):
            with open(roas_path, 'rb') as f:
                combined_roas = pickle.load(f)
        elif os.path.exists(old_roas_path):
            with open(old_roas_path, 'rb') as f:
                combined_roas = pickle.load(f)

        return {
            'box_map': box_map,
            'morse_graph': morse_graph,
            'basins': basins,
            'combined_morse_graph': combined_morse_graph,
            'combined_roas': combined_roas,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Warning: Failed to load cache for method {method_num}: {e}")
        return None


def save_shared_data(output_dir: str,
                    trajectories: list,
                    X: np.ndarray,
                    Y: np.ndarray,
                    gp_model: Optional[Any] = None) -> None:
    """
    Save shared data used by multiple methods.

    Creates directory structure:
        {output_dir}/computation/shared/
            trajectories.pkl
            X.npy
            Y.npy
            gp_model.pkl (optional)

    :param output_dir: Output directory
    :param trajectories: List of trajectory arrays
    :param X: Initial states (N, D)
    :param Y: States after tau (N, D)
    :param gp_model: Trained GP model (optional, for method 4)
    """
    shared_dir = os.path.join(output_dir, 'computation', 'shared')
    os.makedirs(shared_dir, exist_ok=True)

    # Save trajectories with pickle
    with open(os.path.join(shared_dir, 'trajectories.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)

    # Save numpy arrays
    np.save(os.path.join(shared_dir, 'X.npy'), X)
    np.save(os.path.join(shared_dir, 'Y.npy'), Y)

    # Save GP model if provided
    if gp_model is not None:
        with open(os.path.join(shared_dir, 'gp_model.pkl'), 'wb') as f:
            pickle.dump(gp_model, f)


def load_shared_data(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load shared data used by multiple methods.

    :param output_dir: Output directory
    :return: Dict with keys: trajectories, X, Y, gp_model
             Returns None if cache doesn't exist
    """
    shared_dir = os.path.join(output_dir, 'computation', 'shared')

    # Check if required files exist
    if not os.path.exists(os.path.join(shared_dir, 'trajectories.pkl')):
        return None
    if not os.path.exists(os.path.join(shared_dir, 'X.npy')):
        return None
    if not os.path.exists(os.path.join(shared_dir, 'Y.npy')):
        return None

    try:
        # Load trajectories
        with open(os.path.join(shared_dir, 'trajectories.pkl'), 'rb') as f:
            trajectories = pickle.load(f)

        # Load numpy arrays
        X = np.load(os.path.join(shared_dir, 'X.npy'))
        Y = np.load(os.path.join(shared_dir, 'Y.npy'))

        # Load GP model if exists
        gp_model = None
        gp_path = os.path.join(shared_dir, 'gp_model.pkl')
        if os.path.exists(gp_path):
            with open(gp_path, 'rb') as f:
                gp_model = pickle.load(f)

        return {
            'trajectories': trajectories,
            'X': X,
            'Y': Y,
            'gp_model': gp_model
        }
    except Exception as e:
        print(f"Warning: Failed to load shared cache: {e}")
        return None


def cache_exists(output_dir: str, method_num: Optional[int] = None) -> bool:
    """
    Check if cache exists for a method or shared data.

    :param output_dir: Output directory
    :param method_num: Method number (1-5), or None for shared data
    :return: True if cache exists
    """
    if method_num is None:
        # Check shared data
        shared_dir = os.path.join(output_dir, 'computation', 'shared')
        return (os.path.exists(os.path.join(shared_dir, 'trajectories.pkl')) and
                os.path.exists(os.path.join(shared_dir, 'X.npy')) and
                os.path.exists(os.path.join(shared_dir, 'Y.npy')))
    else:
        # Check method cache (only require core files, not basins/roas)
        method_dir = os.path.join(output_dir, 'computation', f'method{method_num}')
        required_files = ['box_map.pkl', 'morse_graph.pkl', 'combined_morse_graph.pkl', 'metadata.json']
        return all(os.path.exists(os.path.join(method_dir, fname)) for fname in required_files)


def clear_cache(output_dir: str, method_nums: Optional[list] = None) -> None:
    """
    Clear cached data.

    :param output_dir: Output directory
    :param method_nums: List of method numbers to clear (1-5), or None to clear all
    """
    import shutil

    if method_nums is None:
        # Clear entire computation directory
        comp_dir = os.path.join(output_dir, 'computation')
        if os.path.exists(comp_dir):
            shutil.rmtree(comp_dir)
            print(f"Cleared all cached data from {comp_dir}")
    else:
        # Clear specific methods
        for method_num in method_nums:
            method_dir = os.path.join(output_dir, 'computation', f'method{method_num}')
            if os.path.exists(method_dir):
                shutil.rmtree(method_dir)
                print(f"Cleared cache for method {method_num}")
