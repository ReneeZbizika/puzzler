import os
import sys
import pygame
import random
import json  # Add import for JSON handling

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
show_piece_ids = False  # Flag to toggle piece ID display
current_pieces_folder = ""  # Track the current pieces folder

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
        
        # Draw piece ID if show_piece_ids is enabled
        if show_piece_ids:
            font = pygame.font.SysFont(None, 24)
            id_text = font.render(f"{self.original_index}", True, (255, 0, 0))
            # Position the text in the center of the piece
            text_x = self.current_pos[0] + (self.image.get_width() // 2) - (id_text.get_width() // 2)
            text_y = self.current_pos[1] + (self.image.get_height() // 2) - (id_text.get_height() // 2)
            surface.blit(id_text, (text_x, text_y))

def set_puzzle_dimensions(image_name):    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(project_root, "params", f"{image_name}_params.txt")
        
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith("Scaled dimensions:"):
                dims = line.strip().split(': ')[1]
                width, height = dims.split('x')
                return int(width)*3.76, int(height)*3.76

def load_puzzle_pieces(pieces_folder):
    """Load puzzle pieces from the specified folder with random positions only"""
    global pieces, BOX_WIDTH, BOX_HEIGHT, current_pieces_folder
    
    # Store the current pieces folder name for later use
    current_pieces_folder = pieces_folder
    
    # Extract image name from folder (e.g., "img_1" from "pieces_img_1")
    image_name = pieces_folder.replace("pieces_", "")
    
    pieces = []
    
    # Load image parameters and set box dimensions
    BOX_WIDTH, BOX_HEIGHT = set_puzzle_dimensions(image_name)
    
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
                int(piece_file.split('_')[1].split('.')[0]),  # Use actual piece number from filename
                [start_x, start_y]  # Random start position
            )
            pieces.append(piece)
            
        except Exception as e:
            print(f"Error loading piece {piece_file}: {e}")
    
    print(f"Successfully loaded {len(pieces)} pieces")

# New function to save current piece positions
def save_current_positions():
    """Save the current positions of all puzzle pieces directly to puzzle_centroids_img_X.json"""
    global pieces, current_pieces_folder, save_message, save_timer
    
    try:
        # Create a list to store piece data
        piece_data = []
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Add each piece's current position
        for i, piece in enumerate(pieces):
            # Calculate the center point (centroid) of the piece
            center_x = piece.rect.x + (piece.image.get_width() // 2)
            center_y = piece.rect.y + (piece.image.get_height() // 2)
            
            piece_data.append({
                "id": f"piece_{piece.original_index}",
                "centroid": {
                    "x": center_x,
                    "y": center_y
                }
            })
        
        # Extract the original image name from the current pieces folder
        original_image = current_pieces_folder.replace("pieces_", "")
        
        # Create the data structure for JSON
        data = {"pieces": piece_data}
        
        # Ensure data directory exists
        data_dir = os.path.join(project_root, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Write directly to puzzle_centroids_img_X.json file
        json_path = os.path.join(project_root, "data", f"puzzle_centroids_{original_image}.json")
        
        # Check if file exists and is writeable
        if os.path.exists(json_path):
            # If file exists, ensure it's writeable
            if not os.access(json_path, os.W_OK):
                # Try to make the file writeable
                os.chmod(json_path, 0o666)  # Read/write permissions for everyone
                
        # Write the data to the file (overwriting any existing file)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to physical storage
        
        # Display save confirmation message on screen for 3 seconds
        save_message = f"UPDATED: positions of {len(pieces)} pieces in puzzle_centroids_{original_image}.json"
        save_timer = 180  # 3 seconds at 60 FPS
        
        print(f"Successfully updated positions of {len(pieces)} pieces in {json_path}")
        return True
    except Exception as e:
        print(f"ERROR saving piece positions: {e}")
        # Display error message on screen
        save_message = f"ERROR SAVING: {str(e)}"
        save_timer = 180  # 3 seconds at 60 FPS
        return False

def main():
    global selected_piece, dragging, show_piece_ids
    global save_message, save_timer
    
    # Initialize save message variables
    save_message = ""
    save_timer = 0
    
    # Load actual puzzle pieces
    load_puzzle_pieces("pieces_img_6")  # You can change this to the desired pieces folder

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
                # Add keyboard shortcut to save positions (S key)
                elif event.key == pygame.K_s:
                    save_current_positions()
                # Toggle piece ID display with V key
                elif event.key == pygame.K_v:
                    show_piece_ids = not show_piece_ids
                    print(f"Piece ID display {'enabled' if show_piece_ids else 'disabled'}")

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
            
        # Display instructions
        font = pygame.font.SysFont(None, 24)
        instructions = font.render("Press V to toggle piece IDs, S to save positions, ESC to quit", True, (0, 0, 0))
        screen.blit(instructions, (SCREEN_WIDTH // 2 - instructions.get_width() // 2, SCREEN_HEIGHT - 30))
        
        # Display save message if active
        if save_timer > 0:
            font = pygame.font.SysFont(None, 36)
            if "ERROR" in save_message:
                # Red text for errors
                message_text = font.render(save_message, True, (255, 0, 0))
            else:
                # Green text for success
                message_text = font.render(save_message, True, (0, 150, 0))
            screen.blit(message_text, (SCREEN_WIDTH // 2 - message_text.get_width() // 2, 50))
            save_timer -= 1

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()