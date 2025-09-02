#!/usr/bin/env python3

import numpy as np
import time
import copy
import random
import pickle
from typing import List, Tuple, Optional

from srmt.constraints.constraints import ConstraintBase
from srmt.planning_scene.planning_scene import PlanningScene

from ljcmp.planning.constrained_bi_rrt import ConstrainedBiRRT
from ljcmp.planning.motion_trees import MotionTree
from ljcmp.planning.distance_functions import distance_q
from ljcmp.planning.status_description import GrowStatus

from scipy.spatial.transform import Rotation as R


class ObjectIKConstrainedBiRRT(ConstrainedBiRRT):
    """
    Constrained BiRRT planner that samples joint configurations from IK results
    of random object poses. This planner uses pre-computed IK solutions to bias
    sampling toward valid configurations.
    """
    
    def __init__(self, ik_results_file: str, state_dim=3, constraint: ConstraintBase = None, 
                 validity_fn=None, start_region_fn=None, goal_region_fn=None,
                 sample_from_ik_prob=0.8, random_sample_prob=0.2) -> None:
        """
        Initialize ObjectIKConstrainedBiRRT
        
        Args:
            ik_results_file: Path to pickle file containing IK results
            state_dim: Dimension of state space
            constraint: Constraint object
            validity_fn: Validity checking function
            start_region_fn: Function to sample from start region
            goal_region_fn: Function to sample from goal region
            sample_from_ik_prob: Probability of sampling from IK results
            random_sample_prob: Probability of random sampling
        """
        super().__init__(state_dim=state_dim, constraint=constraint, 
                        validity_fn=validity_fn, start_region_fn=start_region_fn, 
                        goal_region_fn=goal_region_fn)
        
        # Load IK results
        self.ik_results = self._load_ik_results(ik_results_file)
        self.all_joint_solutions = []
        self.all_jacobians = []
        self.all_nulls = []
        
        # Extract all solutions from IK results
        self._extract_solutions()
        
        # Sampling probabilities
        self.sample_from_ik_prob = sample_from_ik_prob
        self.random_sample_prob = random_sample_prob
        
        # Ensure probabilities sum to 1
        total_prob = sample_from_ik_prob + random_sample_prob
        self.sample_from_ik_prob /= total_prob
        self.random_sample_prob /= total_prob
        
        print(f"Loaded {len(self.all_joint_solutions)} IK solutions from {len(self.ik_results['joints'])} object positions")
        print(f"Sampling probabilities: IK={self.sample_from_ik_prob:.2f}, Random={self.random_sample_prob:.2f}")
    
    def _load_ik_results(self, filename: str) -> dict:
        """Load IK results from pickle file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load IK results from {filename}: {e}")
    
    def _extract_solutions(self):
        """Extract all joint solutions from IK results"""
        self.all_joint_solutions = []
        self.all_jacobians = []
        self.all_nulls = []
        
        for position_idx in range(len(self.ik_results['joints'])):
            joints = self.ik_results['joints'][position_idx]
            jacobians = self.ik_results['jacobians'][position_idx]
            nulls = self.ik_results['nulls'][position_idx]
            
            for solution_idx in range(len(joints)):
                self.all_joint_solutions.append(np.array(joints[solution_idx]))
                self.all_jacobians.append(np.array(jacobians[solution_idx]))
                self.all_nulls.append(np.array(nulls[solution_idx]))
    
    def random_sample(self):
        """
        Sample a joint configuration either from IK results or randomly
        """
        # Decide sampling strategy
        rand_val = np.random.random()
        
        if rand_val < self.sample_from_ik_prob and len(self.all_joint_solutions) > 0:
            # Sample from IK results
            return self._sample_from_ik_results()
        else:
            # Use parent's random sampling
            return super().random_sample()
    
    def _sample_from_ik_results(self) -> np.ndarray:
        """
        Sample a joint configuration from the IK results
        """
        # Randomly select a solution from IK results
        solution_idx = np.random.randint(0, len(self.all_joint_solutions))
        q = self.all_joint_solutions[solution_idx].copy()
        
        # Add small random perturbation to increase diversity
        perturbation_std = 0.01  # Small perturbation
        q += np.random.normal(0, perturbation_std, q.shape)
        
        # Project to constraint manifold
        r = self.constraint.project(q)
        if r is False:
            # If projection fails, try another solution
            return self._sample_from_ik_results()
        
        # Check bounds
        if np.any(q < self.lb) or np.any(q > self.ub):
            # If out of bounds, try another solution
            return self._sample_from_ik_results()
        
        return self.enforce_bounds(q)
    
    def sample_with_null_space_perturbation(self, base_solution_idx: Optional[int] = None) -> np.ndarray:
        """
        Sample a joint configuration by perturbing an existing solution in its null space
        
        Args:
            base_solution_idx: Index of base solution to perturb. If None, randomly select.
        
        Returns:
            Perturbed joint configuration
        """
        if len(self.all_joint_solutions) == 0:
            return self.random_sample()
        
        # Select base solution
        if base_solution_idx is None:
            base_solution_idx = np.random.randint(0, len(self.all_joint_solutions))
        
        base_q = self.all_joint_solutions[base_solution_idx].copy()
        null_space = self.all_nulls[base_solution_idx]
        
        if null_space.shape[1] == 0:
            # No null space, return base solution
            return base_q
        
        # Generate random coefficients for null space vectors
        null_coeffs = np.random.normal(0, 0.1, null_space.shape[1])
        
        # Perturb in null space
        null_perturbation = null_space @ null_coeffs
        q = base_q + null_perturbation
        
        # Project to constraint manifold
        r = self.constraint.project(q)
        if r is False:
            return self.random_sample()
        
        # Check bounds
        if np.any(q < self.lb) or np.any(q > self.ub):
            return self.random_sample()
        
        return self.enforce_bounds(q)
    
    def sample_from_similar_poses(self, target_q: np.ndarray, num_neighbors: int = 5) -> np.ndarray:
        """
        Sample from IK solutions that are similar to a target configuration
        
        Args:
            target_q: Target joint configuration
            num_neighbors: Number of similar solutions to consider
        
        Returns:
            Sampled joint configuration
        """
        if len(self.all_joint_solutions) == 0:
            return self.random_sample()
        
        # Find similar solutions
        distances = []
        for solution in self.all_joint_solutions:
            dist = distance_q(target_q, solution)
            distances.append(dist)
        
        # Get indices of closest solutions
        closest_indices = np.argsort(distances)[:num_neighbors]
        
        # Randomly select from closest solutions
        selected_idx = np.random.choice(closest_indices)
        q = self.all_joint_solutions[selected_idx].copy()
        
        # Add small perturbation
        perturbation_std = 0.005
        q += np.random.normal(0, perturbation_std, q.shape)
        
        # Project to constraint manifold
        r = self.constraint.project(q)
        if r is False:
            return self.random_sample()
        
        # Check bounds
        if np.any(q < self.lb) or np.any(q > self.ub):
            return self.random_sample()
        
        return self.enforce_bounds(q)
    
    def get_ik_statistics(self) -> dict:
        """
        Get statistics about the loaded IK solutions
        
        Returns:
            Dictionary containing statistics
        """
        if len(self.all_joint_solutions) == 0:
            return {"error": "No IK solutions loaded"}
        
        solutions_per_position = [len(joints) for joints in self.ik_results['joints']]
        
        stats = {
            "total_positions": len(self.ik_results['joints']),
            "total_solutions": len(self.all_joint_solutions),
            "avg_solutions_per_position": np.mean(solutions_per_position),
            "min_solutions_per_position": np.min(solutions_per_position),
            "max_solutions_per_position": np.max(solutions_per_position),
            "joint_dimension": len(self.all_joint_solutions[0]),
            "robot_config": self.ik_results.get('robot_config', 'unknown')
        }
        
        return stats 