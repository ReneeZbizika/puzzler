import copy
import numpy as np
import os


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
    
# --- State Class ---
class State:
    def __init__(self, pieces, assembly, unplaced_pieces, edge_info):
        """
        pieces: dict mapping piece_id to Piece objects (used for rendering and simulation)
        assembly: a 2D array (or any structure) representing the current puzzle assembly
        unplaced_pieces: a list (or set) of piece IDs not yet placed
        edge_info: any additional data about piece compatibility
        """
        self.pieces = pieces # dict mapping piece_id -> Piece object
        self.assembly = assembly
        self.unplaced_pieces = unplaced_pieces
        self.edge_info = edge_info
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
        new_edge_info = copy.deepcopy(self.edge_info)
        new_state = State(new_pieces, new_assembly, new_unplaced, new_edge_info)
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
    def __init__(self, piece_id, delta_x, delta_y):
        """
        Represents an action that moves a piece.
        piece_id: ID of the piece to move.
        delta_x, delta_y: displacement.
        orientation_change: (optional) change in orientation.
        """
        self.piece_id = piece_id
        self.delta_x = delta_x
        self.delta_y = delta_y
        #self.orientation_change = orientation_change
    
    def __repr__(self):
        # dθ={self.orientation_change}
        return f"Action(piece_id={self.piece_id}, dx={self.delta_x}, dy={self.delta_y})"
    
    
# ----- Environment Dynamics -----
def apply_action(state, action):    #i.e, Transitions
    new_state = state.copy()  # Make a copy for safety in search
    if action.piece_id in new_state.pieces:
        piece = new_state.pieces[action.piece_id]
        piece.update_position(action.delta_x, action.delta_y)
        #piece.orientation += action.orientation_change
    # Optionally update assembly or unplaced_pieces if the piece is placed.
    if action.piece_id in new_state.unplaced_pieces:
        new_state.unplaced_pieces.remove(action.piece_id)
    new_state.update_edge_info()
    new_state.time_elapsed += 1  # or whatever time increment you use
    return new_state

# ----- Helper Functions for Game State (W/o Render) -----
# Why is 3.8 for scaled dims?
def parse_params_file(image_name):
    """
    Reads the parameter file for the given image and extracts:
      - scaled_dimensions as a tuple (width, height)
      - grid as a tuple (rows, columns)
      - total_pieces as an integer
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(project_root, "params", f"{image_name}_params.txt")
    
    scaled_dimensions = None
    grid = None
    total_pieces = None

    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Scaled dimensions:"):
                dims = line.split(': ')[1]      # "136x98"
                width, height = dims.split('x')
                scaled_dimensions = (int(width) * 3.8, int(height) * 3.8)
            elif line.startswith("Grid:"):
                grid_str = line.split(': ')[1]    # "5x5"
                rows, cols = grid_str.split('x')
                grid = (int(rows), int(cols))
            elif line.startswith("Total pieces:"):
                total_pieces = int(line.split(': ')[1])
    
    return scaled_dimensions, grid, total_pieces

#TODO: make image_name dynamic hardcoding
image_name = "img_2"

scaled_dimensions, grid, total_pieces = parse_params_file(image_name)
puzzle_width, puzzle_height = scaled_dimensions[0], scaled_dimensions[1]
# Define margins
margin_x, margin_y = 100, 100

# Set screen dimensions accordingly
SCREEN_WIDTH = int(puzzle_width) * 1.5 + 2 * margin_x
SCREEN_HEIGHT = int(puzzle_height) * 1.5 + 2 * margin_y

# The solution box is exactly the puzzle dimensions
BOX_WIDTH = puzzle_width
BOX_HEIGHT = puzzle_height

# Center the solution box on the screen
BOX_X = (SCREEN_WIDTH - BOX_WIDTH) // 2
BOX_Y = (SCREEN_HEIGHT - BOX_HEIGHT) // 2

num_rows = grid[0]
num_cols = grid[1]
cell_width = BOX_WIDTH / num_cols
cell_height = BOX_HEIGHT / num_rows


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
    for piece in state.unplaced_pieces:
        for movement in possible_moves():       # e.g., candidate movement vectors
            #for layer_op in possible_layer_ops():    # e.g., operations affecting layering
            #   action = Action(piece.id, movement, layer_op)
            action = Action(piece.id, movement)
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
    for piece in state.pieces.values():
        if not (BOX_X <= piece.x <= BOX_X + BOX_WIDTH - piece.image.get_width() and 
                BOX_Y <= piece.y <= BOX_Y + BOX_HEIGHT - piece.image.get_height()):
            return False
    return True
    """
    # naive solution, move all of them once
    if state.unplaced_pieces == []:
        return True
    else:
        return False
    



