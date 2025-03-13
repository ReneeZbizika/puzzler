import copy
import numpy as np


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

