import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from .dynamics import Dynamics
from .grids import AbstractGrid


class Model:
    """
    The core that connects dynamics and grids.
    """

    def __init__(self, grid: AbstractGrid, dynamics: Dynamics,
                 dynamics_kwargs: dict = None):
        """
        :param grid: The grid that discretizes the state space.
        :param dynamics: The dynamical system.
        :param dynamics_kwargs: Additional kwargs to pass to dynamics.__call__
                               (optional, usually not needed)
        """
        self.grid = grid
        self.F = dynamics
        self.dynamics_kwargs = dynamics_kwargs or {}

    def compute_box_map(self, n_jobs: int = -1) -> nx.DiGraph:
        """
        Compute the BoxMap.

        For each active box, computes its image under the dynamics and converts
        it to grid box indices. Note that for BoxMapData with output_enclosure='box_enclosure'
        (the default), grid.box_to_indices() creates a FILLED RECTANGULAR REGION of boxes,
        not just a sparse union. This implements the "cubical convex closure" of the output.

        :param n_jobs: The number of jobs to run in parallel. -1 means using all
                       available CPUs.
        :return: A directed graph representing the BoxMap, where nodes are box
                 indices and edges represent possible transitions.
        """
        boxes = self.grid.get_boxes()
        active_box_indices = self.F.get_active_boxes(self.grid)

        # Check if grid supports batch mode for box_to_indices
        has_batch_mode = hasattr(self.grid, 'box_to_indices_batch')
        
        if has_batch_mode:
            # Batch mode: compute all image boxes in parallel, then process indices in batch
            def compute_image_box(i):
                return i, self.F(boxes[i], **self.dynamics_kwargs)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_image_box)(i) for i in active_box_indices
            )
            
            # Extract indices and image boxes
            indices = [i for i, _ in results]
            image_boxes = np.array([img_box for _, img_box in results])
            
            # Use batch mode to compute all adjacencies at once
            adjacencies = self.grid.box_to_indices_batch(image_boxes)
            
            # Build graph from batch results
            graph = nx.DiGraph()
            graph.add_nodes_from(active_box_indices)
            
            for i, adj in zip(indices, adjacencies):
                for j in adj:
                    if j in active_box_indices:
                        graph.add_edge(i, j)
        else:
            # Standard mode: compute adjacencies individually in parallel
            def compute_adjacencies(i):
                image_box = self.F(boxes[i], **self.dynamics_kwargs)
                # Note: grid.box_to_indices() finds ALL boxes that intersect image_box.
                # For output_enclosure='box_enclosure', this creates a filled rectangle.
                adj = self.grid.box_to_indices(image_box)
                return i, adj

            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_adjacencies)(i) for i in active_box_indices
            )

            graph = nx.DiGraph()
            # Only add active boxes as nodes
            graph.add_nodes_from(active_box_indices)

            for i, adj in results:
                for j in adj:
                    # Only add edges to active boxes (j might be inactive)
                    if j in active_box_indices:
                        graph.add_edge(i, j)

        return graph