import copy
import numpy as np
import os
import pygame
import random
import time
import json

# --- Filepath definitions ---
#TODO: make image_name dynamic hardcoding
image_name = "img_2"

# Build the file path to your JSON file.
#TODO change img_name to dynamic 
params_folder = "params"
img_name = "img_2"  # or derive from your original image path
param_file_path = os.path.join(params_folder, f"{img_name}_params.json")
#pieces_path = os.path.join(f"pieces_{img_name}") TODO - change this to puzzle_pieces/pieces_{image_name}

root_eval = "Datasets/evaluation"
#root = "Datasets"
pieces_folder = f"pieces_{img_name}"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pieces_path = os.path.join(project_root, pieces_folder)


# --- Piece Class ---
class Piece:
    def __init__(self, piece_id, image, x, y):
        self.id = piece_id        # Unique identifier for the piece.
        self.image = image        # Pygame surface or image data.
        self.x = x                # Current x position.
        self.y = y                # Current y position.
        # self.orientation = orientation
        # self.rect = pygame.Rect(x, y, image.get_width(), image.get_height())
    
    def update_position(self, dx, dy):
        """Move the piece by a delta."""
        self.x += dx
        self.y += dy
        # self.rect.x = self.x
        # self.rect.y = self.y
    
    def set_position(self, x, y):
        """Set the piece's position explicitly."""
        self.x = x
        self.y = y
    
    def copy(self):
        # Shallow copy is usually sufficient if image data is read-only.
        #return Piece(self.id, self.image, self.x, self.y, self.orientation)
        return Piece(self.id, self.image, self.x, self.y)
    
    # if using self.rect, make sure NOT to convert to torch tensor
    # self.rect = pygame.Rect(x, y, image.get_width(), image.get_height())
    # used in terminal state in game_agent
    
    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))
    
    """    
    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))
    """
    
"""def load_puzzle_pieces(pieces_folder):
    #Load puzzle pieces from the specified folder with random initial positions.
    
    global BOX_WIDTH, BOX_HEIGHT

    pieces_dict = {}
    
    # Load image parameters and set box dimensions
    #BOX_WIDTH, BOX_HEIGHT = set_puzzle_dimensions("img_2")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces_path = os.path.join(project_root, pieces_folder)
    
    if not os.path.exists(pieces_path):
        print(f"Error: Pieces folder '{pieces_path}' not found!")
        return {}
    
    piece_files = [f for f in os.listdir(pieces_path) if f.endswith('.png')]
    if not piece_files:
        print(f"Error: No PNG files found in '{pieces_path}'")
        return {}
    
    piece_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"Loading {len(piece_files)} puzzle pieces")
    
    start_x_range = (50, SCREEN_WIDTH - 150)
    start_y_range = (50, SCREEN_HEIGHT - 150)
    
    for i, piece_file in enumerate(piece_files):
        try:
            piece_path = os.path.join(pieces_path, piece_file)
            image = pygame.image.load(piece_path).convert_alpha()
            
            # Generate a random position
            start_x = random.randint(*start_x_range)
            start_y = random.randint(*start_y_range)
            # Re-generate if the position is inside the grey rectangle
            while (BOX_X - 5 <= start_x <= BOX_X + BOX_WIDTH + 5) and (BOX_Y - 5 <= start_y <= BOX_Y + BOX_HEIGHT + 5):
                start_x = random.randint(*start_x_range)
                start_y = random.randint(*start_y_range)
            
            piece = Piece(piece_id=i+1, image=image, x=start_x, y=start_y)
            pieces_dict[piece.id] = piece
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
    print(f"Successfully loaded {len(pieces_dict)} pieces")
    return pieces_dict"""

def load_puzzle_pieces(pieces_folder):
    """
    Load puzzle pieces from the specified folder with random initial positions.
    
    Returns:
        dict: Mapping from piece_id to Piece objects.
    """
    pieces_dict = {}
    
    # Determine the absolute path to the pieces folder.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces_path = os.path.join(project_root, pieces_folder)
    
    if not os.path.exists(pieces_path):
        print(f"Error: Pieces folder '{pieces_path}' not found!")
        return {}
    
    piece_files = [f for f in os.listdir(pieces_path) if f.lower().endswith('.png')]
    if not piece_files:
        print(f"Error: No PNG files found in '{pieces_path}'")
        return {}
    
    # Sort files assuming filenames like "piece_1.png", "piece_2.png", etc.
    piece_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"Loading {len(piece_files)} puzzle pieces")
    
    start_x_range = (50, SCREEN_WIDTH - 150)
    start_y_range = (50, SCREEN_HEIGHT - 150)
    
    for i, piece_file in enumerate(piece_files):
        try:
            piece_path = os.path.join(pieces_path, piece_file)
            image = pygame.image.load(piece_path).convert_alpha()
            
            # Generate a random starting position.
            start_x = random.randint(*start_x_range)
            start_y = random.randint(*start_y_range)
            # Regenerate if the position falls within (or too close to) the grey solution box.
            while (BOX_X - 5 <= start_x <= BOX_X + BOX_WIDTH + 5) and (BOX_Y - 5 <= start_y <= BOX_Y + BOX_HEIGHT + 5):
                start_x = random.randint(*start_x_range)
                start_y = random.randint(*start_y_range)
            
            piece = Piece(piece_id=i+1, image=image, x=start_x, y=start_y)
            pieces_dict[piece.id] = piece
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
    print(f"Successfully loaded {len(pieces_dict)} pieces")
    return pieces_dict
    
    
# --- State Class ---
class State:
    def __init__(self, pieces, assembly, unplaced_pieces): #TODO: removed edge_info for now, discuss with group
        """
        pieces: dict mapping piece_id to Piece objects (used for rendering and simulation)
        assembly: a 2D array (or any structure) representing the current puzzle assembly
        unplaced_pieces: a list (or set) of piece IDs not yet placed
        *edge_info: any additional data about piece compatibility - unused for now
        """
        self.pieces = pieces # dict mapping piece_id -> Piece object
        self.assembly = assembly
        #self.unplaced_pieces = set(pieces.keys())
        #TODO change this from list to set
        self.unplaced_pieces = unplaced_pieces
        #self.edge_info = edge_info
        self.time_elapsed = 0  # Can track steps or elapsed time
    
    #TODO
    def update_edge_info(self):
        # Update edge compatibility info based on current assembly.
        pass
    
    def copy(self):
        # Create a deep copy of the state for simulation purposes.
        new_pieces = {pid: piece.copy() for pid, piece in self.pieces.items()}
        new_assembly = np.copy(self.assembly)  # Assuming assembly is a NumPy array.
        new_unplaced = self.unplaced_pieces.copy()
        #new_edge_info = copy.deepcopy(self.edge_info)
        new_state = State(new_pieces, new_assembly, new_unplaced)
        new_state.time_elapsed = self.time_elapsed
        return new_state
    
    """
    def copy(self):
        # Create a shallow copy of the state (deep copy each Piece if needed)
        new_pieces = {}
        for pid, piece in self.pieces.items():
            # Assuming image can be shared; copy positions and rects
            new_piece = Piece(piece.id, piece.image, piece.x, piece.y)
            new_pieces[pid] = new_piece
        new_state = State(new_pieces)
        new_state.time_elapsed = self.time_elapsed
        return new_state
    """

# --- Action Class ---
class Action:
    def __init__(self, piece_id, dx, dy):
        """
        Represents an action that moves a piece.
        piece_id: ID of the piece to move.
        delta_x, delta_y: displacement.
        orientation_change: (optional) change in orientation.
        """
        self.piece_id = piece_id
        self.dx = dx
        self.dy = dy
        #self.orientation_change = orientation_change
    
    def __repr__(self):
        # dθ={self.orientation_change}
        return f"Action(piece_id={self.piece_id}, dx={self.dx}, dy={self.dy})"
    
    
# ----- Environment Dynamics -----
def apply_action(state, action):    #i.e, Transitions
    new_state = state.copy()  # Make a copy for safety in search
    if action.piece_id in new_state.pieces:
        piece = new_state.pieces[action.piece_id]
        piece.update_position(action.dx, action.dy)
        #piece.orientation += action.orientation_change
    # Optionally update assembly or unplaced_pieces if the piece is placed.
    if action.piece_id in new_state.unplaced_pieces:
        new_state.unplaced_pieces.remove(action.piece_id)
    #new_state.update_edge_info()
    new_state.time_elapsed += 1  # or whatever time increment you use
    return new_state

# ----- Helper Functions for Game State (W/o Render) -----

# Load the JSON file.
with open(param_file_path, 'r') as f:
    data = json.load(f)


# ----------- Constants from image_params.JSON -----------
# Now you can grab any value from the JSON.
image_path = data["Image"]
original_dims = data["Original dimensions"]
scaled_dims = data["Scaled dimensions"]
scale_factor = data["Scale factor"]
grid = data["Grid"]
xn = data["Num Row Pieces"]
yn = data["Num Col Pieces"]
total_pieces = data["Total pieces"]
BOX_WIDTH = data["BOX_WIDTH"]
BOX_HEIGHT = data["BOX_HEIGHT"]
SCREEN_WIDTH = data["SCREEN_WIDTH"]
SCREEN_HEIGHT = data["SCREEN_HEIGHT"]
command = data["Command"]
BG_COLOR = (240, 240, 240)
BOX_COLOR = (180, 180, 180)

#print("Image:", image_path)
#print("Original dimensions:", original_dims)
#print("Scaled dimensions:", scaled_dims)
#print("Grid:", grid)
#print("Command:", command)

# Center the solution box on the screen
# Override in game_agent, where we set to 50, 50
BOX_X = 50
BOX_Y = 50
#BOX_X = (SCREEN_WIDTH - BOX_WIDTH) // 2
#BOX_Y = (SCREEN_HEIGHT - BOX_HEIGHT) // 2

cell_width = BOX_WIDTH / xn
cell_height = BOX_HEIGHT / yn

# Dummy placeholder functions (#TODO: have to implement this)
#TODO
def possible_moves(state):
    """
    Return candidate moves as offsets. For example, moves that shift a piece by one cell in any direction.
    This includes cardinal and diagonal moves.
    """
    # Offsets in terms of grid cells
    grid_moves = [(-1, 0), (1, 0), (0, -1), (0, 1),  # cardinal directions
                  (-1, -1), (1, 1), (-1, 1), (1, -1)] # diagonal moves

    # Convert cell moves to pixel moves:
    candidate_moves = [(int(dx * cell_width), int(dy * cell_height)) for dx, dy in grid_moves]
    return candidate_moves

def valid_actions(state):
    # Return a list of valid actions for the state
    # return []
    """
    Generate a list of valid actions from the given state.
    - Valid actions: 
    - Puzzle pieces are UNPLACED
    - Puzzle action is within frame
    Perform early pruning based on edge compatibility.
    """
    actions = []
    for piece_id, _ in state.pieces.items():
        for movement in possible_moves(state):       # e.g., candidate movement vectors
            #for layer_op in possible_layer_ops():    # e.g., operations affecting layering
            #   action = Action(piece.id, movement, layer_op)
            action = Action(piece_id, movement[0], movement[1])
            # Lightweight early pruning: only add if compatibility passes a threshold
            # SKIP FOR NOW - dont want actor to have priviledged info about edges
            # if compute_edge_compatibility(state, action) >= COMPATIBILITY_THRESHOLD:
            actions.append(action)
    return actions

# Dummy placeholder functions (#TODO: have to implement this)
#TODO: change so that it is not dependant on the PyGame functions
def is_terminal(state):
    """
    Define a terminal condition. For example, when all pieces are
    inside the puzzle box (you can check if their positions lie within BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT).
    """
    # piece.rect.width , piece.rect.height
    """
    if not(state.unplaced_pieces):
        return True
    else:
        return False
    """
    # naive solution, move all of them once
    # state.unplaced_piece is a set
    if valid_actions(state) == None:
        return True
    # Check if all pieces are within the puzzle box
    for piece in state.pieces.values():
        if not (BOX_X <= piece.x <= BOX_X + BOX_WIDTH - piece.image.get_width() and 
                BOX_Y <= piece.y <= BOX_Y + BOX_HEIGHT - piece.image.get_height()):
            return False
    return True
    
    
# --- Environment Functions (Can be wrapped with Gymnasium) ---
def create_initial_assembly():
    """
    Creates an empty puzzle assembly matrix.
    Adjust dimensions based on puzzle size.
    """
    puzzle_size = (xn, yn)  # use xn and yn from JSON
    return np.zeros(puzzle_size, dtype=np.int32)  # Empty puzzle representation

#TODO - a different load_puzzle_pieces
"""
def load_puzzle_pieces():
 
    Loads puzzle pieces (could be from a dataset or predefined list).
    Each piece might have an ID, edges, and other properties.

    # total_pieces from JSON
    return [{"id": i, "edges": None} for i in range(1, total_pieces + 1)]
"""

def initialize_edge_info():
    """
    Initializes edge compatibility information.
    Can be computed based on image analysis or loaded from a dataset.
    """
    return np.random.rand(25, 4)  # Example: 25 pieces with 4 edge values each

#TODO
def reset():
        # Properly initialize and return a State object.
        state = initialize_state(pieces_folder, xn, yn)
        return state

def initialize_state(pieces_folder, grid_rows, grid_cols):
    """
    Initialize the puzzle state.

    Parameters:
      - pieces_folder: Folder path containing puzzle piece images (or data).
      - grid_rows: Number of rows in the puzzle grid.
      - grid_cols: Number of columns in the puzzle grid.

    Returns:
      A State object with:
        - pieces: Loaded pieces (dict mapping piece_id to Piece objects).
        - assembly: A 2D NumPy array (of zeros) representing the initial empty assembly.
        - unplaced_pieces: A set of all piece IDs (since nothing is placed initially).
    """
    # Load pieces from the folder. Assume load_puzzle_pieces returns a dict {piece_id: Piece, ...}
    pieces = load_puzzle_pieces(pieces_folder)
    
    # Create an assembly array. Adjust the shape as needed.
    assembly = np.zeros((grid_rows, grid_cols))
    
    # Initially, all pieces are unplaced. Create a set of all piece IDs.
    unplaced_pieces = set(pieces.keys())
    
    # Create and return the initial state.
    state = State(pieces, assembly, unplaced_pieces)
    return state

# Define dimensions based on environment data
def get_dimensions():
    """
    Dynamically determines:
    - state_dim: The length of the flattened state vector.
    - action_dim: The total number of possible actions.
    - visual_dim: The number of extracted visual features.
    
    Returns:
        state_dim (int), action_dim (int), visual_dim (int)
    """
    # Initialize a sample state from the environment
    sample_state = initialize_state(pieces_path, xn, yn)

    # Compute state dimension: Flattened representation of the state
    state_dim = 30  # Assuming 'assembly' is a NumPy array

    # Compute action dimension: Count the total number of valid actions for a sample state
    sample_actions = valid_actions(sample_state)
    action_dim = xn* yn * len(sample_actions) if sample_actions else xn * yn * 8  # Fallback if empty

    # Compute visual feature dimension: Adjust based on extracted visual features
    visual_dim = 2  # Example, could be edge matching, SSIM similarity, etc.
    #print(state_dim, action_dim, visual_dim)
    return state_dim, action_dim, visual_dim

# ----- Rendering Functions -----
def render_state(screen, state):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, BOX_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))
    for piece in state.pieces.values():
        piece.draw(screen)
    pygame.display.flip()

def step(state, action):
    """
    Execute an action in the given state and return the next state, reward, done flag, and info.
    
    Parameters:
        state: The current state
        action: The action to take
        
    Returns:
        next_state: The new state after taking the action
        reward: The reward received for taking the action
        done: Boolean indicating if the episode is finished
        info: Dictionary containing additional information
    """
    # Apply the action to get the next state
    next_state = apply_action(state, action)
    
    # Calculate reward (you can use the compute_intermediate_reward function from mcts.py)
    # For now, use a simple reward
    reward = 0.0
    if hasattr(next_state, 'pieces') and hasattr(action, 'piece_id'):
        # Check if the piece is in the correct position
        # This is a simplified version - you might want to use your existing reward function
        reward = 1.0 if is_piece_correctly_placed(next_state, action.piece_id) else -0.1
    
    # Check if the episode is done
    done = is_terminal(next_state)
    
    # Additional info
    info = {}
    
    return next_state, reward, done, info

def is_piece_correctly_placed(state, piece_id):
    """
    Simple helper function to check if a piece is correctly placed.
    You might want to replace this with your existing logic.
    """
    # Placeholder implementation - replace with your actual logic
    return False