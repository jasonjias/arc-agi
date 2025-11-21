import json
import os.path
import numpy as np
from ArcAgent import find_enclosed_rings

# Load the problem
problem_name = "4b6b68e5.json"
milestone_path = os.path.join('Milestones', 'D')

with open(os.path.join(milestone_path, problem_name)) as p:
    flat_data = json.load(p)
    test_input = np.array(flat_data['test'][0]['input'])

rings = find_enclosed_rings(test_input, bg=0)

for ring_color, region_cells in rings:
    if ring_color == 8:
        print(f"Gray ring (color 8):")

        # Simulate the boundary finding logic
        h, w = test_input.shape
        boundary_seeds = set()
        for r, c in region_cells:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and test_input[nr, nc] == ring_color:
                    boundary_seeds.add((nr, nc))

        print(f"  Boundary seeds: {len(boundary_seeds)} cells")
        print(f"  Is (15, 9) a boundary seed? {(15, 9) in boundary_seeds}")

        # Flood fill from boundary
        boundary_component = set()
        stack = list(boundary_seeds)
        visited = set(boundary_seeds)

        while stack:
            r, c = stack.pop()
            boundary_component.add((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and test_input[nr, nc] == ring_color:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

        print(f"  Boundary component: {len(boundary_component)} cells")
        print(f"  Is (15, 9) in boundary? {(15, 9) in boundary_component}")

        # Find interior pieces
        all_ring_pixels = set(zip(*np.where(test_input == ring_color)))
        interior_pieces = all_ring_pixels - boundary_component

        print(f"  Total ring pixels: {len(all_ring_pixels)}")
        print(f"  Interior pieces: {len(interior_pieces)} cells")
        print(f"  Is (15, 9) an interior piece? {(15, 9) in interior_pieces}")

        if interior_pieces:
            print(f"  Interior piece coordinates: {sorted(interior_pieces)}")
