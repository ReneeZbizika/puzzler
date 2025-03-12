import os
import sys
import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (240, 240, 240)
BOARD_COLOR = (180, 180, 180)
BOX_X = 50
BOX_Y = 50

# Global variable for board dimensions (set later)
BOX_WIDTH = 600  
BOX_HEIGHT = 700

# ----- Piece Class -----
class Piece:
    def __init__(self, piece_id, image, x, y):
        self.id = piece_id
        self.image = image
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, image.get_width(), image.get_height())

    def update_position(self, dx, dy):
        self.x += dx
        self.y += dy
        self.rect.x = self.x
        self.rect.y = self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))

# ----- State Class -----
class State:
    def __init__(self, pieces):
        # pieces: dictionary mapping piece_id -> Piece object
        self.pieces = pieces  # all puzzle pieces
        self.time_elapsed = 0  # can track time or steps

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

# ----- Action Class -----
class Action:
    def __init__(self, piece_id, delta_x, delta_y):
        self.piece_id = piece_id
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __repr__(self):
        return f"Action(piece_id={self.piece_id}, dx={self.delta_x}, dy={self.delta_y})"

# ----- Helper Functions for Loading Pieces -----
def set_puzzle_dimensions(image_name):    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(project_root, "params", f"{image_name}_params.txt")
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith("Scaled dimensions:"):
                dims = line.strip().split(': ')[1]
                width, height = dims.split('x')
                return int(width)*3.8, int(height)*3.8

def load_puzzle_pieces(pieces_folder):
    """Load puzzle pieces from the specified folder with random initial positions."""
    global BOX_WIDTH, BOX_HEIGHT

    pieces_dict = {}
    
    # Load image parameters and set box dimensions
    BOX_WIDTH, BOX_HEIGHT = set_puzzle_dimensions("img_2")
    
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
            start_x = random.randint(*start_x_range)
            start_y = random.randint(*start_y_range)
            piece = Piece(piece_id=i+1, image=image, x=start_x, y=start_y)
            pieces_dict[piece.id] = piece
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
    
    print(f"Successfully loaded {len(pieces_dict)} pieces")
    return pieces_dict

# ----- Rendering Functions -----
def render_state(screen, state):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, BOARD_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))
    for piece in state.pieces.values():
        piece.draw(screen)
    pygame.display.flip()

# ----- Environment Dynamics -----
def apply_action(state, action):
    """Apply an action to the state and return a new state."""
    new_state = state.copy()
    if action.piece_id in new_state.pieces:
        piece = new_state.pieces[action.piece_id]
        piece.update_position(action.delta_x, action.delta_y)
    # Update time, etc.
    new_state.time_elapsed += 1
    return new_state

def is_terminal(state):
    """
    Define a terminal condition. For example, when all pieces are
    inside the puzzle box (you can check if their positions lie within BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT).
    """
    for piece in state.pieces.values():
        if not (BOX_X <= piece.x <= BOX_X + BOX_WIDTH - piece.rect.width and 
                BOX_Y <= piece.y <= BOX_Y + BOX_HEIGHT - piece.rect.height):
            return False
    return True

# ----- Placeholder for Agent Action Selection (e.g., via MCTS) -----
def agent_select_action(state):
    """
    This function should use MCTS (or any agent logic) to choose the next action.
    For now, we provide a dummy version that randomly moves one piece.
    """
    # Pick a random piece from state
    piece_id = random.choice(list(state.pieces.keys()))
    # For example, move by a small random delta (you can define your own action space)
    delta_x = random.randint(-20, 20)
    delta_y = random.randint(-20, 20)
    action = Action(piece_id, delta_x, delta_y)
    print("Agent selected:", action)
    return action

# ----- Main Game Loop (Agent-Controlled Version) -----
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jigsaw Puzzle Game")
    font = pygame.font.SysFont(None, 36) # Initialize a font (using the default font, size 36)
    
    # Instead of interactive dragging, load the pieces and create a State object.
    pieces_dict = load_puzzle_pieces("pieces_img_2")
    if not pieces_dict:
        return
    
    state = State(pieces_dict)
    clock = pygame.time.Clock()
    running = True

    while running:
        # Instead of handling mouse events, let the agent choose an action
        if not is_terminal(state):
            action = agent_select_action(state)
            state = apply_action(state, action)
        else:
            print("Terminal state reached!")
            running = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Render the current state
        render_state(screen, state)
        
        # Render the elapsed time display with nicer formatting
        # Create a panel background for the text
        panel_width = 200
        panel_height = 40
        panel_color = (240, 240, 240)  # Light gray
        panel_border = (50, 50, 50)    # Dark gray
        # Position the panel in the top-right corner with some margin
        panel_x = SCREEN_WIDTH - panel_width - 20
        panel_y = 20
        # Draw the panel with rounded corners and border
        pygame.draw.rect(screen, panel_border, (panel_x-2, panel_y-2, panel_width+4, panel_height+4), border_radius=10)
        pygame.draw.rect(screen, panel_color, (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        # Format the moves text with a nice font
        try:
            # Try to use a nicer font if available
            time_font = pygame.font.Font(None, 28)  # None uses default font, 28 is size
        except:
            # Fall back to system font if custom font fails
            time_font = font
        # Format the text with better labels
        time_text = time_font.render(f"Time Elapsed: {state.time_elapsed}", True, (20, 20, 20))
        # Center the text in the panel
        text_x = panel_x + (panel_width - time_text.get_width()) // 2
        text_y = panel_y + (panel_height - time_text.get_height()) // 2
        # Draw the text
        screen.blit(time_text, (text_x, text_y))
        
        pygame.display.flip()
        
        # For visualization, wait a short period between actions (e.g., 500 ms)
        pygame.time.wait(500)
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
