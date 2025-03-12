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
BOX_COLOR = (180, 180, 180)  # Grey puzzle area
GRID_COLOR = (200, 200, 200)
SOLVED_BORDER_COLOR = (0, 255, 0)  # Green border for solved puzzles

# Puzzle board area
BOX_WIDTH = 600  # Half the screen
BOX_HEIGHT = 700
BOX_X = 50
BOX_Y = 50

# Game state
pieces = []
selected_piece = None
dragging = False
game_complete = False

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Jigsaw Puzzle Game")

class PuzzlePiece:
    def __init__(self, image, original_index, target_pos, current_pos):
        self.image = image
        self.original_index = original_index
        self.target_pos = target_pos
        self.current_pos = current_pos
        self.rect = pygame.Rect(self.current_pos[0], self.current_pos[1], image.get_width(), image.get_height())
        self.is_selected = False
        self.placed_correctly = False

    def update_position(self, pos):
        self.current_pos = list(pos)
        self.rect.x = pos[0]
        self.rect.y = pos[1]

    def check_correct_placement(self):
        """Check if the piece is inside its correct box area"""
        x_correct = abs(self.rect.x - self.target_pos[0]) < 20
        y_correct = abs(self.rect.y - self.target_pos[1]) < 20
        if x_correct and y_correct:
            self.rect.x, self.rect.y = self.target_pos  # Snap into position
            self.placed_correctly = True
        return self.placed_correctly

    def draw(self, surface):
        surface.blit(self.image, self.current_pos)
        if self.placed_correctly:
            pygame.draw.rect(surface, SOLVED_BORDER_COLOR, self.rect, 3)  # Green border for correct pieces

def load_puzzle_pieces(pieces_folder, additional_scale=1.0):
    """Load puzzle pieces from the specified folder"""
    global pieces, BOX_WIDTH, BOX_HEIGHT
    pieces = []
    
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
    
    # Calculate grid dimensions based on number of pieces
    total_pieces = len(piece_files)
    rows = 5  # Default rows
    cols = total_pieces // rows
    if total_pieces % rows != 0:
        cols += 1
    
    print(f"Loading {total_pieces} puzzle pieces in a {rows}x{cols} grid")
    
    # First pass: load all pieces
    piece_surfaces = []
    
    for piece_file in piece_files:
        try:
            # Load the image with transparency
            piece_path = os.path.join(pieces_path, piece_file)
            surface = pygame.image.load(piece_path).convert_alpha()
            
            # Apply additional scaling if needed
            if additional_scale != 1.0:
                original_width = surface.get_width()
                original_height = surface.get_height()
                new_width = int(original_width * additional_scale)
                new_height = int(original_height * additional_scale)
                surface = pygame.transform.smoothscale(surface, (new_width, new_height))
            
            piece_surfaces.append(surface)
            
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
            piece_surfaces.append(None)  # Placeholder for failed loads
    
    # For a perfect fit, we need to know the dimensions of the original image
    # Since pieces may have different sizes, we need to use the original template dimensions
    
    # We can get this from the full_pipeline.py output or from the template.png file
    # For now, let's estimate it based on the pieces we have
    
    # We know the pieces were generated from a grid, so we can estimate
    # the original dimensions by looking at the piece positions in the grid
    
    # For a 5x5 grid, we need to find the rightmost edge of column 4 and
    # the bottom edge of row 4 (0-indexed)
    
    # Starting positions for pieces
    start_x_range = (BOX_X + BOX_WIDTH + 50, SCREEN_WIDTH - 150)
    start_y_range = (50, SCREEN_HEIGHT - 150)
    
    # Create puzzle pieces with proper positions
    # We'll use the original_index to determine the correct grid position
    pieces_by_position = {}
    
    for i, surface in enumerate(piece_surfaces):
        if surface is None:
            continue
            
        # Get piece number from filename
        try:
            piece_num = int(piece_files[i].split('_')[1].split('.')[0]) - 1  # 0-indexed
        except (IndexError, ValueError):
            piece_num = i
        
        # Calculate grid position
        row = piece_num // cols
        col = piece_num % cols
        
        # Store the piece info by its grid position
        pieces_by_position[(row, col)] = {
            'surface': surface,
            'width': surface.get_width(),
            'height': surface.get_height(),
            'index': piece_num + 1  # Back to 1-indexed for display
        }
    
    # Now we need to calculate the exact position for each piece in the solved puzzle
    # We'll do this by working out the positions from left to right, top to bottom
    
    # First, calculate the width of each column
    column_widths = [0] * cols
    for (row, col), piece_info in pieces_by_position.items():
        column_widths[col] = max(column_widths[col], piece_info['width'])
    
    # Then, calculate the height of each row
    row_heights = [0] * rows
    for (row, col), piece_info in pieces_by_position.items():
        row_heights[row] = max(row_heights[row], piece_info['height'])
    
    # Calculate the total board dimensions
    ideal_board_width = sum(column_widths)
    ideal_board_height = sum(row_heights)
    
    # Update the global board dimensions
    BOX_WIDTH = ideal_board_width
    BOX_HEIGHT = ideal_board_height
    print(f"Setting puzzle board size to: {BOX_WIDTH}x{BOX_HEIGHT}")
    
    # Calculate the starting position for each column
    column_starts = [0] * cols
    for i in range(1, cols):
        column_starts[i] = column_starts[i-1] + column_widths[i-1]
    
    # Calculate the starting position for each row
    row_starts = [0] * rows
    for i in range(1, rows):
        row_starts[i] = row_starts[i-1] + row_heights[i-1]
    
    # Now create the actual puzzle pieces with correct target positions
    for (row, col), piece_info in pieces_by_position.items():
        # Calculate the target position
        target_x = BOX_X + column_starts[col]
        target_y = BOX_Y + row_starts[row]
            
        # Random start position (outside the box)
        start_x = random.randint(*start_x_range)
        start_y = random.randint(*start_y_range)

        # Create the puzzle piece
        piece = PuzzlePiece(
            piece_info['surface'], 
            piece_info['index'], 
            [target_x, target_y], 
            [start_x, start_y]
        )
        pieces.append(piece)
        print(f"Positioned piece {piece_info['index']} at grid ({row},{col}): {target_x},{target_y}")
    
    print(f"Successfully loaded {len(pieces)} pieces")

def check_game_complete():
    """Check if all puzzle pieces are correctly placed."""
    return all(piece.placed_correctly for piece in pieces)

def main():
    global selected_piece, dragging, game_complete
    
    # Load actual puzzle pieces instead of generating random ones
    load_puzzle_pieces("pieces_img_2")
    
    # If no pieces were loaded, fall back to generating random ones
    if not pieces:
        print("No pieces loaded, generating random pieces instead")
        generate_puzzle_pieces()

    clock = pygame.time.Clock()
    running = True
    offset_x, offset_y = 0, 0

    while running:
        screen.fill(BG_COLOR)

        # Draw puzzle box
        pygame.draw.rect(screen, BOX_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))

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
                    if selected_piece:
                        selected_piece.check_correct_placement()
                        selected_piece = None

            elif event.type == pygame.MOUSEMOTION:
                if dragging and selected_piece:
                    selected_piece.update_position((event.pos[0] + offset_x, event.pos[1] + offset_y))

        # Draw puzzle pieces
        for piece in pieces:
            piece.draw(screen)

        # Check if game is complete
        game_complete = check_game_complete()
        if game_complete:
            font = pygame.font.SysFont('Arial', 36)
            text = font.render('Puzzle Complete!', True, (0, 200, 0))
            screen.blit(text, (SCREEN_WIDTH//2 - 100, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

# Keep the generate_puzzle_pieces function as a fallback
def generate_puzzle_pieces():
    """Generate puzzle pieces on one side of the screen"""
    global pieces
    pieces = []
    
    rows, cols = 5, 6  # Define puzzle grid
    piece_size = 100  # Each puzzle piece size
    
    # Starting positions (randomized in the right half of the screen)
    start_x_range = (BOX_X + BOX_WIDTH + 50, SCREEN_WIDTH - piece_size - 50)
    start_y_range = (50, SCREEN_HEIGHT - piece_size - 50)
    
    # Target positions (inside the grey box)
    grid_x_offset = BOX_X + 50
    grid_y_offset = BOX_Y + 50
    
    for row in range(rows):
        for col in range(cols):
            # Compute correct target position inside the grey box
            target_x = grid_x_offset + col * piece_size
            target_y = grid_y_offset + row * piece_size
            
            # Random start position (outside the box)
            start_x = random.randint(*start_x_range)
            start_y = random.randint(*start_y_range)

            # Create a placeholder piece (colored square for now)
            piece_surface = pygame.Surface((piece_size, piece_size))
            piece_surface.fill((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

            # Create puzzle piece object
            piece = PuzzlePiece(piece_surface, len(pieces), [target_x, target_y], [start_x, start_y])
            pieces.append(piece)

if __name__ == "__main__":
    main()
