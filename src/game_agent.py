import os
import sys
import pygame
import random
import time
import numpy as np

from env import State, Action, Piece, apply_action
from mcts import MCTS # Import your MCTS function from your MCTS module
# from mcts import apply_action as mcts_apply_action, is_terminal

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


def is_terminal(state):
    """
    Define a terminal condition. For example, when all pieces are
    inside the puzzle box (you can check if their positions lie within BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT).
    """
    # piece.rect.width , iece.rect.height
    for piece in state.pieces.values():
        if not (BOX_X <= piece.x <= BOX_X + BOX_WIDTH - piece.image.get_width() and 
                BOX_Y <= piece.y <= BOX_Y + BOX_HEIGHT - piece.image.get_height()):
            return False
    return True


# ----- [DUMMY] Agent Action Selection -----
def dummy_agent_select_action(state):
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

# ----- Agent Action Selection (via MCTS) -----
def agent_select_action(state):
    """
    Use MCTS to select the next action given the current state.
    This function calls the MCTS routine (which uses your policy/value networks) 
    to search the tree and returns the best action.
    """
    # Run MCTS for a fixed number of iterations (e.g., 100)
    action = MCTS(state, iterations=100)
    print("Agent selected action:", action)
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
    
    # Define an assembly matrix for the puzzle. For example, a 5x6 grid (adjust dimensions as needed):
    assembly = np.zeros((5, 6))
    
    # The unplaced pieces are all the pieces that were loaded:
    unplaced_pieces = list(pieces_dict.keys())
    
    #TODO update edge_info using edge_match.py
    # Start with an empty edge info dictionary (or load your specific edge data)
    edge_info = {}
    
    # Create the State object using the unified State class from env.py
    state = State(pieces_dict, assembly, unplaced_pieces, edge_info)
    
    clock = pygame.time.Clock()
    running = True

    while running:
        # Instead of handling mouse events, let the agent choose an action
        if not is_terminal(state):
            action = dummy_agent_select_action(state) #switch between dummy and real
            #action = dummy_agent_select_action(state)
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
        
        # Render the elapsed time text on the screen
        #TODO: currently just displays number of seconds (ex. 100) make it look nice
        time_text = font.render(f"Time Elapsed: {state.time_elapsed}", True, (0, 0, 0))
        screen.blit(time_text, (10, 10))
        
        pygame.display.flip()
        
        # For visualization, wait a short period between actions (e.g., 500 ms)
        pygame.time.wait(500)
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
