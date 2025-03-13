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

from env import State, Action, Piece, apply_action, valid_actions, is_terminal, possible_moves
from models import PolicyNetwork, ValueNetwork

# Constants (set these appropriately)
MAX_SIM_DEPTH = 10
TIME_PER_MOVE = 1.0
MCTS_ITERATIONS = 100
COMPATIBILITY_THRESHOLD = 0.5
C = 1.0  # Exploration constant

# You should set these dimensions based on your state representation.
state_dim = 100  
action_dim = 10

# Create the models (initially untrained)
policy_model = PolicyNetwork(state_dim, action_dim)
value_model = ValueNetwork(state_dim)


# --- Neural Network Helper Functions (Using PyTorch) ---
def convert_state_to_tensor(state):
    """
    Convert your state into a flat tensor of size (1, state_dim).
    """
     # Example: assume state has a list of piece positions stored in state.assembly (a 2D NumPy array)
     # only grab numerical stuff
    flat_state = state.assembly.flatten() 
    # Ensure it is a float tensor and add a batch dimension.
    state_tensor = torch.tensor(flat_state, dtype=torch.float32).unsqueeze(0)
    return state_tensor

def policy_network_forward(state):
    """
    Convert the state into a tensor representation and pass it through the PyTorch policy model.
    Returns a probability distribution over valid actions.
    """
    state_tensor = convert_state_to_tensor(state)  # user-defined function
    probs = policy_model(state_tensor)             # policy_model: a pretrained PyTorch model, shape: (1, action_dim)
    return probs

def value_network_forward(state):
    """
    Convert the state into a tensor and return the predicted state value.
    """
    state_tensor = convert_state_to_tensor(state)
    value = value_model(state_tensor)              # value_model: a pretrained PyTorch model
    return value.item() #return value.item or return value?


# --- MCTS Core Functions ---
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state      # The state at this node
        self.parent = parent    # Parent node in the search tree
        self.action = action    # Action taken from parent to reach this state
        self.children = []      # List of child nodes
        self.visits = 0         # Number of visits
        self.total_reward = 0   # Total reward accumulated

def selection(node):
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
            pi = 1.0 / len(node.children)
            # pi = policy_network_forward(child.parent.state)[child.action]  # pseudo-access; adjust as needed
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
        return node  # No expansion if terminal
    actions = valid_actions(node.state)
    for action in actions:
        new_state = apply_action(node.state, action)
        child_node = Node(state=new_state, parent=node, action=action)
        node.children.append(child_node)
    # Optionally, return one of the new child nodes for simulation
    return random.choice(node.children) if node.children else node

def simulation(state, max_depth=MAX_SIM_DEPTH):
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
        policy_probs = policy_network_forward(current_state)  # Shape: (1, action_dim)
        
        # For each valid action, map it to an index (make sure your action_to_index function is consistent).
        valid_action_probs = []
        for action in actions:
            idx = action_to_index(action)
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
    # approximate future rewards that the simulation didn’t cover.
    if not is_terminal(current_state):
        cumulative_reward += value_network_forward(current_state)
    return cumulative_reward

def backpropagation(node, reward):
    """
    Propagate the simulation result back up the tree.
    """
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent
        
        
#TODO edit render_fn = render_state, edit redit_state from mcts.py or game_agent.py (should be only one function, unite)
def MCTS(root_state, iterations=100, render=False, render_fn=None):
    root = Node(root_state)
    for i in range(iterations):
        leaf = selection(root)
        expanded = expansion(leaf)
        reward = simulation(expanded.state)
        backpropagation(expanded, reward)
        
        # If rendering is enabled, call the rendering function with the current state.
        if render and render_fn is not None:
            render_fn(root_state)  # or pass expanded.state as desired.
    best_child = max(root.children, key=lambda child: child.visits)
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
#TODO add params for just basic assembly
def compute_intermediate_reward(state, action, time_penalty, mode='visual'):
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
    if mode == 'visual':
        compatibility_score = compute_edge_compatibility(state, action)
        visual_score = evaluate_visual(state, action)
        # weight_edge and weight_visual should be defined globally or passed as parameters
        reward = (weight_edge * compatibility_score +
                  weight_visual * visual_score -
                  time_penalty)
        return reward
    elif mode == 'assembly':
        # For example, check if the piece lands in its correct cell.
        # You might have a helper function that returns True/False if piece is correctly assembled.
        if is_piece_correctly_assembled(state, action.piece_id):
            # Give a positive reward if correctly assembled (you can tune this value)
            reward = 1.0 - time_penalty
        else:
            # Otherwise, provide a negative or zero reward (penalize the time penalty)
            reward = -time_penalty
        return reward
    else:
        raise ValueError("Invalid mode. Choose 'visual' or 'assembly'.")


# --- Conversions ---
#TODO allow for unfixed candidate moves
def action_to_index(action):
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
    candidate_moves = [(-5, 0), (-4, 0), (-3, 0), 
                       (0, -5), (0, -4), (0, -3),
                       (3, 0),  (4, 0),  (5, 0),
                       (0, 3),  (0, 4),  (0, 5)]
    
    try:
        move_index = candidate_moves.index((action.dx, action.dy))
    except ValueError:
        raise ValueError(f"Movement vector {(action.dx, action.dy)} is not in the candidate moves list!")
    
    # Convert piece_id (assumed to be 1-indexed) to zero-based index.
    piece_index = action.piece_id - 1
    
    # Calculate the overall index.
    index = piece_index * len(candidate_moves) + move_index
    return index


#TODO evaluate edges function
# --- Reward and Evaluation Functions ---
def compute_edge_compatibility(state, action):
    """
    Evaluate how well the edges of the piece selected by 'action'
    will match with its neighbors in the current 'state'.
    Returns a score (e.g., between 0 and 1).
    """
    # Could use differential invariant signatures, compatibility matrix, etc.
    score = evaluate_edges(state, action)  # user-defined evaluation function
    return score


# --- Environment Functions (Can be wrapped with Gymnasium) ---
def initialize_state():
    """
    Set up the initial state of the puzzle.
    """
    assembly = create_initial_assembly()          # a zero matrix with dimensions based on puzzle specs
    unplaced_pieces = load_puzzle_pieces()          # load puzzle pieces
    edge_info = initialize_edge_info()              # compute or load initial edge compatibility info
    return State(assembly, unplaced_pieces, edge_info)

# --- PyGame Rendering ---
#TODO implement rendering and pass to game_v2
"""
def render_state(screen, state):
    # Draw background, box, etc.
    for pid, piece in state.pieces.items():
        # If orientation is non-zero, rotate piece.image on the fly
        screen.blit(piece.image, (piece.x, piece.y))
    pygame.display.flip()
"""

# Entry point
#if __name__ == "__main__":
#    main()
