#!/usr/bin/env python3

import pickle
import numpy as np
import argparse

def load_ik_results(filename):
    """Load IK results from pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_results(data):
    """Analyze the loaded IK results"""
    print("=" * 60)
    print("IK Results Analysis")
    print("=" * 60)
    
    # Basic info
    print(f"Robot Configuration: {data['robot_config']}")
    print(f"Generation Parameters:")
    for key, value in data['generation_parameters'].items():
        print(f"  {key}: {value}")
    
    # Data structure analysis
    print(f"\nData Structure:")
    print(f"  Joints: {len(data['joints'])} positions")
    print(f"  Jacobians: {len(data['jacobians'])} positions")
    print(f"  Nulls: {len(data['nulls'])} positions")
    
    # Solutions analysis
    total_solutions = 0
    solutions_per_position = []
    
    for i, joints in enumerate(data['joints']):
        num_solutions = len(joints)
        total_solutions += num_solutions
        solutions_per_position.append(num_solutions)
        print(f"  Position {i+1}: {num_solutions} solutions")
    
    print(f"\nTotal Solutions: {total_solutions}")
    print(f"Average Solutions per Position: {np.mean(solutions_per_position):.2f}")
    print(f"Min Solutions per Position: {np.min(solutions_per_position)}")
    print(f"Max Solutions per Position: {np.max(solutions_per_position)}")
    
    # Joint data analysis
    if len(data['joints']) > 0 and len(data['joints'][0]) > 0:
        first_solution = data['joints'][0][0]
        print(f"\nJoint Configuration:")
        print(f"  Number of joints: {len(first_solution)}")
        print(f"  First solution: {first_solution}")
    
    # Jacobian analysis
    if len(data['jacobians']) > 0 and len(data['jacobians'][0]) > 0:
        first_jacobian = np.array(data['jacobians'][0][0])
        print(f"\nJacobian Analysis:")
        print(f"  Jacobian shape: {first_jacobian.shape}")
        print(f"  Rank: {np.linalg.matrix_rank(first_jacobian)}")
    
    # Null space analysis
    if len(data['nulls']) > 0 and len(data['nulls'][0]) > 0:
        first_null = np.array(data['nulls'][0][0])
        print(f"\nNull Space Analysis:")
        print(f"  Null space shape: {first_null.shape}")
        print(f"  Null space dimension: {first_null.shape[1]}")

def main():
    parser = argparse.ArgumentParser(description='Load and analyze IK results')
    parser.add_argument('filename', type=str, help='Path to the IK results pickle file')
    parser.add_argument('--show-solutions', action='store_true', help='Show detailed solution data')
    
    args = parser.parse_args()
    
    try:
        # Load the data
        print(f"Loading IK results from: {args.filename}")
        data = load_ik_results(args.filename)
        
        # Analyze the results
        analyze_results(data)
        
        # Show detailed solutions if requested
        if args.show_solutions:
            print("\n" + "=" * 60)
            print("Detailed Solutions")
            print("=" * 60)
            
            for i, (joints, jacobians, nulls) in enumerate(zip(data['joints'], data['jacobians'], data['nulls'])):
                print(f"\nPosition {i+1}:")
                for j, (joint, jac, null) in enumerate(zip(joints, jacobians, nulls)):
                    print(f"  Solution {j+1}:")
                    print(f"    Joints: {joint}")
                    print(f"    Jacobian shape: {np.array(jac).shape}")
                    print(f"    Null space shape: {np.array(null).shape}")
                    if j >= 2:  # Limit output to first 3 solutions per position
                        print(f"    ... (showing first 3 solutions)")
                        break
        
        return data
        
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

if __name__ == "__main__":
    main() 