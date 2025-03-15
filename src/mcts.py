# Pseudocode Outline
import os
import torch
import numpy
import collections
import random
import math
import pdb
import time
import argparse
import json

from env import State, Action, Piece, apply_action, valid_actions, is_terminal, possible_moves, centroids_folder
from env import initialize_state
from models import PolicyNetwork, ValueNetwork
from env import get_dimensions

#TODO: move paths into a seperate yaml file
from env import image_name

from utils_features import evaluate_assembly_compatibility, extract_visual_features
#evaluate_assembly_compatibility(assembly_state, edge_to_piece_map, edge_compatibility)

# Constants (set these appropriately)
MAX_SIM_DEPTH = 5 # change from 10 to 2
TIME_PER_MOVE = 0.01
MCTS_ITERATIONS = 5  # Instead of 100, change to 5
COMPATIBILITY_THRESHOLD = 0.5
C = 0.25  # Exploration constant

# --- Neural Network Helper Functions (Using PyTorch) ---
def convert_state_to_tensor(state):
    """
    Convert your state into a flat tensor of size (1, state_dim).
    """
    # Create a feature vector with the correct size (30 features)
    state_vector = []
    
    # Extract features from the state
    if hasattr(state, 'assembly') and state.assembly is not None:
        flat_state = state.assembly.flatten()
        state_vector.extend(flat_state.tolist())
    
    # Extract features from pieces if available
    if hasattr(state, 'pieces'):
        # Handle pieces as a dictionary or list
        if isinstance(state.pieces, dict):
            pieces_list = list(state.pieces.values())
        else:
            pieces_list = state.pieces
            
        for piece in pieces_list:
            if hasattr(piece, 'x') and hasattr(piece, 'y'):
                # Normalize position values
                x = piece.x / 1200.0  # Assuming max width is 1200
                y = piece.y / 800.0   # Assuming max height is 800
                state_vector.extend([x, y])
    
    # Ensure we have exactly 30 features
    if len(state_vector) < 30:
        # Pad with zeros if we have fewer than 30 features
        state_vector.extend([0.0] * (30 - len(state_vector)))
    else:
        # Truncate if we have more than 30 features
        state_vector = state_vector[:30]
    
    # Convert to tensor and add batch dimension
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
    
    return state_tensor

def policy_network_forward(current_state, policy_model):
    """Forward pass through the policy network"""
    # Create a feature vector from the state
    state_vector = []
    
    # Extract features from pieces if available
    if hasattr(current_state, 'pieces'):
        # Handle pieces as a list
        if isinstance(current_state.pieces, list):
            for piece in current_state.pieces:
                if hasattr(piece, 'current_pos'):
                    # Normalize position values
                    x = piece.current_pos[0] / 1200.0
                    y = piece.current_pos[1] / 800.0
                    state_vector.extend([x, y])
    
    # Ensure we have exactly 30 features
    if len(state_vector) < 30:
        # Pad with zeros if we have fewer than 30 features
        state_vector.extend([0.0] * (30 - len(state_vector)))
    else:
        # Truncate if we have more than 30 features
        state_vector = state_vector[:30]
    
    # Convert to tensor and add batch dimension
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
    
    # Forward pass through the model
    probs = policy_model(state_tensor)
    return probs

# visual_features: 
# Edge compatibility between adjacent pieces.
# Current assembled image similarity to the final solution.
# Histogram comparison or SSIM scores between assembled and target images.
#  can compute these features inside env.py using OpenCV or other methods.

# Extract visual features; this returns a list, e.g. [similarity_score, edge_score]
def value_network_forward(state, value_model):
    """
    Convert the state into a tensor and return the predicted state value.
    Visual features (e.g., img similarity and edge compatibility scores) are passed as a second input.
    """
    state_tensor = convert_state_to_tensor(state)
    
    # Extract visual features and ensure it has the correct shape (2 features)
    try:
        visual_features_raw = extract_visual_features(state, image_name)
        
        # Handle the case where extract_visual_features returns a single value
        if isinstance(visual_features_raw, (float, numpy.float64, numpy.float32)):
            # If it's a single value, create a list with two elements
            visual_features = [float(visual_features_raw), 0.0]
        elif hasattr(visual_features_raw, '__iter__'):
            # If it's already an iterable (list, tuple, array), convert to list
            visual_features = list(visual_features_raw)
            # Make sure we have exactly 2 features
            if len(visual_features) < 2:
                # If we have fewer than 2 features, pad with zeros
                visual_features = visual_features + [0.0] * (2 - len(visual_features))
            elif len(visual_features) > 2:
                # If we have more than 2 features, truncate
                visual_features = visual_features[:2]
        else:
            # Fallback for any other type
            visual_features = [0.0, 0.0]
            
    except Exception as e:
        # Fallback if feature extraction fails
        print(f"Warning: Visual feature extraction failed: {e}")
        visual_features = [0.0, 0.0]  # Default values
    
    visual_tensor = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0)
    
    # Ensure visual_tensor has shape (1, 2)
    assert visual_tensor.shape == (1, 2), f"Visual tensor has incorrect shape: {visual_tensor.shape}, expected (1, 2)"
    
    value = value_model(state_tensor, visual_tensor)  # Updated to pass both inputs
    
    return value.item()


# --- MCTS Core Functions ---
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state      # The state at this node
        self.parent = parent    # Parent node in the search tree
        self.action = action    # Action taken from parent to reach this state
        self.children = []      # List of child nodes
        self.visits = 0         # Number of visits
        self.total_reward = 0   # Total reward accumulated

def selection(node, policy_model):
    """
    Traverse the tree starting at 'node' using a variant of PUCT until a leaf is reached.
    Uses the policy network for prior probabilities and a UCB-like formula.
    """
    while node.children:
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            # UCB formula with policy prior: Q + C * pi * sqrt(N_parent) / (1 + N_child)
            # For illustration, we assume a uniform prior (or you could use policy_network)
            # pi = 1.0 / len(node.children)
            # Convert child's action to an integer index.
            idx = action_to_index(child.parent.state, child.action)
            pi = policy_network_forward(child.parent.state, policy_model)[0, idx].item()
            #pi = policy_network_forward(child.parent.state, policy_model)[child.action]  # pseudo-access; adjust as needed
            ucb_score = (child.total_reward / (child.visits + 1e-5) +
                         C * pi * (math.sqrt(node.visits) / (1 + child.visits)))
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        node = best_child
    return node

def expansion(node):
    """
    Expand the leaf node by adding all valid child nodes.
    """
    if is_terminal(node.state):
        # Don't print this message since we'll show expansion results on main line
        return node  # No expansion if terminal
    
    actions = valid_actions(node.state)
    
    # Clear existing children (if any)
    node.children = []
    
    if not actions:
        # Just a simple warning, but won't add extra line
        print(f"[WARNING: No valid actions]", end=" ")
        return node
    
    for action in actions:
        new_state = apply_action(node.state, action)
        child_node = Node(state=new_state, parent=node, action=action)
        node.children.append(child_node)
    
    # Return a random child node for simulation
    if node.children:
        chosen_child = random.choice(node.children)
        return chosen_child
    else:
        return node  # Return the original node if no children were created

# Pass visual features into value_network_forward() for evaluation.
def simulation(state, policy_model, value_model, max_depth=MAX_SIM_DEPTH):
    """
    Simulate the outcome starting from 'state' until a terminal state or depth cutoff.
    At each step, use the policy network to sample an action (the actor) and compute the intermediate reward.
    If the simulation doesn't reach a terminal state, use the value network (the critic)
    to approximate future rewards.
    """
    cumulative_reward = 0
    current_state = state
    depth = 0
    while (not is_terminal(current_state)) and (depth < max_depth):
        actions = valid_actions(current_state)
        if not actions:
            break
        
        # Get the probability distribution from the policy network.
        policy_probs = policy_network_forward(current_state, policy_model)  # Shape: (1, action_dim)
        
        # For each valid action, map it to an index (make sure your action_to_index function is consistent).
        valid_action_probs = []
        for action in actions:
            idx = action_to_index(state, action)
            # Get the probability from the network. We assume the network outputs probabilities in a tensor.
            valid_action_probs.append(policy_probs[0, idx].item())
            
        # Normalize the probabilities (in case they don't sum exactly to 1).
        total_prob = sum(valid_action_probs)
        if total_prob == 0:
            # Fallback: if the network gives zero probabilities, choose randomly.
            chosen_action = random.choice(actions)
        else:
            normalized_probs = [p / total_prob for p in valid_action_probs]
            # Sample an action using the normalized probabilities.
            chosen_action = random.choices(actions, weights=normalized_probs, k=1)[0]

        # Compute the immediate (intermediate) reward for the chosen action.
        intermediate_reward = compute_intermediate_reward(current_state, chosen_action, TIME_PER_MOVE)
        cumulative_reward += intermediate_reward
        
        # Update the state.
        current_state = apply_action(current_state, chosen_action)
        depth += 1
        
    # If simulation ended before reaching a terminal state, boost using the value network
    # approximate future rewards that the simulation didn't cover.
    if not is_terminal(current_state):
        cumulative_reward += value_network_forward(current_state, value_model)
    
    return cumulative_reward

def backpropagation(node, reward):
    """
    Propagate the simulation result back up the tree.
    """
    node_count = 0
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent
        node_count += 1
    
    # Optionally return the number of nodes updated
    return node_count

#TODO edit render_fn = render_state, edit redit_state from mcts.py or game_agent.py (should be only one function, unite)
def MCTS(root_state, policy_model, value_model, iterations=100, render=False, render_fn=None):
    root = Node(root_state)
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"[STARTING MCTS: {iterations} ITERATIONS]")
    print(f"{'='*50}")
    
    for i in range(iterations):
        # Print for every iteration instead of every 5
        elapsed = time.time() - start_time
        print(f"\n[ITERATION {i+1}/{iterations}] [Time: {elapsed:.2f}s]", end=" ")
        
        # Selection phase
        leaf = selection(root, policy_model)
        print(f"[Selection]", end=" ")
        
        # Count children before expansion
        children_before = len(leaf.children)
        
        # Expansion phase
        expanded = expansion(leaf)
        
        # Count children after expansion to get the actual number created
        if expanded == leaf:
            num_new_children = len(leaf.children) - children_before
        else:
            # If a different node was returned, it's a child node
            num_new_children = len(leaf.children)
        
        print(f"[Expansion: {num_new_children} children]", end=" ")
        
        # Simulation phase
        reward = simulation(expanded.state, policy_model, value_model)
        print(f"[Reward: {reward:.4f}]", end=" ")
        
        # Backpropagation phase
        nodes_updated = backpropagation(expanded, reward)
        print(f"[Backpropagation: {nodes_updated} nodes]")
        
        # If rendering is enabled, call the rendering function with the current state.
        if render and render_fn is not None:
            render_fn(root_state)
    
    # Print summary statistics
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"[MCTS SUMMARY]")
    print(f"{'='*50}")
    print(f"[Total Iterations: {iterations}] [Total Time: {total_time:.2f}s] [Avg Time/Iteration: {total_time/iterations:.4f}s]")
    
    # Print information about the children
    if root.children:
        print(f"\n[TOP ACTIONS BY VISIT COUNT]")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children[:5]):  # Show top 5
            print(f"  [{i+1}] [Piece ID: {child.action.piece_id}, dx: {child.action.dx}, dy: {child.action.dy}] " 
                  f"[Visits: {child.visits}] [Avg Reward: {child.total_reward/max(1, child.visits):.4f}]")
    else:
        print("[WARNING] [Root node has no children!]")
    
    # Select best child
    #TODO: how do we want to handle no actions found?
    if not root.children:
        print("[ERROR] [No valid actions found!]")
        return None
        #print("No valid actions found!")
        #print("No valid actions found. Returning a no-op action.")
        #print("No valid actions left; treating this as a terminal state.")
        #done = True
        #return
        #return Action(piece_id=None, dx=0, dy=0)
        #return None
        
    best_child = max(root.children, key=lambda child: child.visits)
    print(f"\n[SELECTED BEST ACTION] [Piece ID: {best_child.action.piece_id}, dx: {best_child.action.dx}, dy: {best_child.action.dy}] "
          f"[Visits: {best_child.visits}] [Avg Reward: {best_child.total_reward/max(1, best_child.visits):.4f}]")
    print(f"{'='*50}")
    
    return best_child.action

# Then, in your Train.py (or wherever you call MCTS), 
# you can pass render=False for training runs, and render=True 
# (plus a proper rendering function, e.g. render_state(screen, state, font)) 
# when you want to visualize.


# Assume possible_movements() returns a list of movement vectors.
# Can either move in any direction with unlimited movement, or we could limit it to 5
# For now we limit it to 5

def evaluate_visual(state, action):
    return 1.0  # dummy value

def update_assembly(assembly, action):
    # Update the assembly matrix.
    return assembly

#TODO add params for visual checking

# Global cache for centroids
_CENTROIDS_CACHE = None

def load_puzzle_centroids(filename=centroids_folder):
    global _CENTROIDS_CACHE
    if _CENTROIDS_CACHE is not None:
        return _CENTROIDS_CACHE
        
    with open(filename, "r") as f:
        data = json.load(f)
    
    centroids = {}
    for piece in data.get("pieces", []):
        pid_str = piece.get("id", "")
        if pid_str.startswith("piece_"):
            try:
                pid = int(pid_str.split("_")[1])
            except ValueError:
                continue
        else:
            continue
        
        centroid = piece.get("centroid", {})
        x = centroid.get("x")
        y = centroid.get("y")
        if x is not None and y is not None:
            centroids[pid] = (x, y)
    
    _CENTROIDS_CACHE = centroids
    return centroids

def is_piece_correctly_assembled(state, piece_id, centroids, tolerance=10):
    """
    Check if the piece with the given piece_id is placed correctly,
    by comparing its current position with the target centroid (from the JSON file).
    
    Parameters:
      - state: The current state (which contains a dictionary state.pieces mapping piece IDs to Piece objects).
      - piece_id: The ID of the piece to check (as an integer).
      - tolerance: Maximum allowed Euclidean distance (in pixels) between the piece's current position and its target.
      
    Returns:
      True if the piece is within tolerance of its target position, False otherwise.
    """
    target = centroids.get(piece_id)
    if target is None:
        raise ValueError(f"No centroid found for piece {piece_id}")
    
    piece = state.pieces.get(piece_id)
    if piece is None:
        raise ValueError(f"Piece {piece_id} not found in state")
    
    current_pos = (piece.x, piece.y)
    dx = current_pos[0] - target[0]
    dy = current_pos[1] - target[1]
    distance = (dx**2 + dy**2) ** 0.5
    
    return distance <= tolerance

"""def generate_candidate_moves(max_dx, max_dy, step=1, exclude_zero=True):
    Generate all integer (dx, dy) movement vectors where:
      - dx is in [-max_dx, max_dx], stepping by `step`
      - dy is in [-max_dy, max_dy], stepping by `step`
    If exclude_zero is True, (0,0) will be omitted (since it doesn't move).
    
    For example:
        generate_candidate_moves(5, 5, step=1)
      returns all (dx, dy) with dx, dy in [-5..5], excluding (0,0).
    
    Parameters:
        max_dx (int): Maximum displacement in the x direction (positive or negative).
        max_dy (int): Maximum displacement in the y direction (positive or negative).
        step (int): Step size for enumerating dx and dy.
        exclude_zero (bool): If True, exclude the (0,0) move.
        
    Returns:
        list of (int, int): The list of all possible candidate movement vectors.
    moves = []
    for dx in range(-max_dx, max_dx + 1, step):
        for dy in range(-max_dy, max_dy + 1, step):
            if exclude_zero and dx == 0 and dy == 0:
                continue
            moves.append((dx, dy))
    return moves"""
#TODO add params for just basic assembly
def compute_intermediate_reward(state, action, time_penalty, mode='assembly'):
    """
    Compute the immediate reward for taking an action.
    Two possible reward schemes:
      1. 'visual': A weighted combination of edge compatibility and visual evaluation.
      2. 'assembly': A simpler reward that checks if the piece is in the correct location.
      
    Parameters:
      - state: The current state.
      - action: The action taken.
      - time_penalty: A penalty for time or moves.
      - mode: 'visual' or 'assembly'.
      
    Returns:
      A reward value (float).
    """
    #if mode == 'visual':
        #compatibility_score = compute_edge_compatibility(state, action)
        #visual_score = evaluate_visual(state, action)
        # weight_edge and weight_visual should be defined globally or passed as parameters
        #reward = (weight_edge * compatibility_score +
                  #weight_visual * visual_score -
                  #time_penalty)
        #return reward
        #pass
    if mode == 'assembly':
        # For example, check if the piece lands in its correct cell.
        # You might have a helper function that returns True/False if piece is correctly assembled.
        
        #TODO: default = Datasets/puzzle_centroids.json for pieces_img_2
        centroids = load_puzzle_centroids() 
        tolerance = 10 #TODO: define this in env.py instead
        
        if is_piece_correctly_assembled(state, action.piece_id, centroids, tolerance): 
            # Give a positive reward if correctly assembled (you can tune this value)
            reward = 50.0 - time_penalty
        else:
            # Otherwise, provide a negative or zero reward (penalize the time penalty)
            reward = -time_penalty  
        return reward
    else:
        raise ValueError("Invalid mode. Choose 'visual' or 'assembly'.")

# --- Conversions ---
#TODO allow for unfixed candidate moves
def action_to_index(state, action):
    """
    Map an action (which contains a piece_id and a movement vector (dx, dy))
    to a unique index in the policy network's output.
    
    We assume that the allowed movement vectors (candidate moves) are fixed.
    For example:
        candidate_moves = [(-5, 0), (-4, 0), (-3, 0), 
                           (0, -5), (0, -4), (0, -3),
                           (3, 0),  (4, 0),  (5, 0),
                           (0, 3),  (0, 4),  (0, 5)]
    
    If there are N candidate moves per piece, and the total number of pieces is P,
    then the policy network should output a vector of length P * N.
    
    The action index is computed as:
       index = (piece_id - 1) * len(candidate_moves) + move_index
    """
    candidate_moves = possible_moves(state)
    #candidate_moves = [(-5, 0), (-4, 0), (-3, 0), 
    #                       (0, -5), (0, -4), (0, -3),
      #                     (3, 0),  (4, 0),  (5, 0),
      #                     (0, 3),  (0, 4),  (0, 5)]
    try:
        move_index = candidate_moves.index((action.dx, action.dy))
    except ValueError:
        raise ValueError(f"Movement vector {(action.dx, action.dy)} is not in the candidate moves list!")
    
    # Convert piece_id (assumed to be 1-indexed) to zero-based index.
    piece_index = action.piece_id - 1
    
    # Calculate the overall index.
    index = piece_index * len(candidate_moves) + move_index
    return index