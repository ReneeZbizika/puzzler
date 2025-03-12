import os
import sys
import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (240, 240, 240)
BOARD_COLOR = (180, 180, 180) 

# Puzzle board area - using fixed dimensions
BOX_X = 50
BOX_Y = 50

# Game state
pieces = []
selected_piece = None
dragging = False

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Jigsaw Puzzle Game")

class PuzzlePiece:
    def __init__(self, image, original_index, current_pos):
        self.image = image
        self.original_index = original_index
        self.current_pos = current_pos
        self.rect = pygame.Rect(self.current_pos[0], self.current_pos[1], image.get_width(), image.get_height())
        self.is_selected = False

    def update_position(self, pos):
        self.current_pos = list(pos)
        self.rect.x = pos[0]
        self.rect.y = pos[1]

    def draw(self, surface):
        surface.blit(self.image, self.current_pos)

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
    """Load puzzle pieces from the specified folder with random positions only"""
    global pieces, BOX_WIDTH, BOX_HEIGHT

    pieces = []
    
    # Load image parameters and set box dimensions
    BOX_WIDTH, BOX_HEIGHT = set_puzzle_dimensions("img_2")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces_path = os.path.join(project_root, pieces_folder)
    
    # Check if the folder exists
    if not os.path.exists(pieces_path):
        print(f"Error: Pieces folder '{pieces_path}' not found!")
        return
    
    # Get all PNG files in the folder
    piece_files = [f for f in os.listdir(pieces_path) if f.endswith('.png')]
    if not piece_files:
        print(f"Error: No PNG files found in '{pieces_path}'")
        return
    
    # Sort the files by their piece number
    piece_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"Loading {len(piece_files)} puzzle pieces")
    
    # Define the area where pieces can be initially placed
    # Random positions around the puzzle area
    start_x_range = (50, SCREEN_WIDTH - 150)
    start_y_range = (50, SCREEN_HEIGHT - 150)
    
    # Load and create puzzle pieces with random positions
    for i, piece_file in enumerate(piece_files):
        try:
            # Load the image with transparency
            piece_path = os.path.join(pieces_path, piece_file)
            surface = pygame.image.load(piece_path).convert_alpha()
            
            # Random start position
            start_x = random.randint(*start_x_range)
            start_y = random.randint(*start_y_range)
            
            # Create the puzzle piece
            piece = PuzzlePiece(
                surface,
                i + 1,  # Use simple index as the piece identifier
                [start_x, start_y]  # Random start position
            )
            pieces.append(piece)
            
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
    
    print(f"Successfully loaded {len(pieces)} pieces")

def main():
    global selected_piece, dragging
    
    # Load actual puzzle pieces
    load_puzzle_pieces("pieces_img_2")

    clock = pygame.time.Clock()
    running = True
    offset_x, offset_y = 0, 0

    while running:
        screen.fill(BG_COLOR)

        # Draw puzzle box
        pygame.draw.rect(screen, BOARD_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for piece in reversed(pieces):  # Check topmost pieces first
                        if piece.rect.collidepoint(event.pos):
                            selected_piece = piece
                            pieces.remove(piece)  # Bring to front
                            pieces.append(piece)
                            dragging = True
                            offset_x = piece.rect.x - event.pos[0]
                            offset_y = piece.rect.y - event.pos[1]
                            break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging and selected_piece:
                    selected_piece.update_position((event.pos[0] + offset_x, event.pos[1] + offset_y))

        # Draw puzzle pieces
        for piece in pieces:
            piece.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()