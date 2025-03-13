import os
import sys
import pygame
import random
import time
import numpy as np

from env import State, Action, Piece
from env import apply_action, is_terminal
#set_puzzle_dimensions
from env import BOX_WIDTH, BOX_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT
from mcts import MCTS # Import your MCTS function from your MCTS module

# Initialize Pygame
pygame.init()
print(BOX_WIDTH, BOX_HEIGHT)
print(SCREEN_WIDTH, SCREEN_HEIGHT)

# Constants
BG_COLOR = (240, 240, 240)
BOARD_COLOR = (180, 180, 180)
BOX_X = 50
BOX_Y = 50

#TODO - use img_path and pieces_path instead of hardcoding 
# Image Filepath 
img_path = "img_2"
pieces_path = f"pieces_{img_path}"

# ----- Helper Functions for Loading Pieces -----
def load_puzzle_pieces(pieces_folder):
    """Load puzzle pieces from the specified folder with random initial positions."""
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
    return pieces_dict

# ----- Rendering Functions -----
def render_state(screen, state):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, BOARD_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))
    for piece in state.pieces.values():
        piece.draw(screen)
    pygame.display.flip()


# ----- [DUMMY] Agent Action Selection -----
def dummy_agent_select_action(state):
    """
    This function should use MCTS (or any agent logic) to choose the next action.
    For now, we provide a dummy verrender_stateon that randomly moves one piece.
    """
    # Pick a random piece from state
    piece_id = random.choice(list(state.pieces.keys()))
    # For example, move by a small random delta (you can define your own action space)
    delta_x = random.randint(-30, 30)
    delta_y = random.randint(-30, 30)
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
    # MCTS(state, iterations=100, render=render_on, render_fn=lambda s: render_state(screen, s))
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
    
    # For training, set render_on to False for speed; set to True if you want visualization.
    render_on = True

    while running:
        # Instead of handling mouse events, let the agent choose an action
        if not is_terminal(state):
            action = dummy_agent_select_action(state) #switch between dummy and real
            #action = agent_select_action(state)
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
                    
        if render_on:
            render_state(screen, state)
            
            #TODO: currently just displays number of seconds (ex. 100) make it look nice
            # also note - time elapsed just is a moves ocunter, doesnt actually track seconds
            time_text = font.render(f"Time Elapsed: {state.time_elapsed}", True, (0, 0, 0))
            screen.blit(time_text, (10, 10))
            pygame.display.flip()
            pygame.time.wait(500)
        
        # For visualization, wait a short period between actions (e.g., 500 ms)
        pygame.time.wait(300)
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
