import os
import sys
import pygame
import random
import math
import tempfile
import subprocess
from PIL import Image
import io

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (255, 255, 0, 100)  # Yellow highlight with alpha
GRID_COLOR = (200, 200, 200)
SOLVED_BORDER_COLOR = (0, 255, 0)  # Green border for solved puzzles

BOX_COLOR = (180, 180, 180)  # Grey puzzle area
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
show_grid = False
show_solution_overlay = False

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Jigsaw Puzzle Game")

class PuzzlePiece:
    def __init__(self, image, original_index, target_pos, current_pos=None):
        self.image = image
        self.original_index = original_index  # The original file number
        self.target_pos = target_pos  # Where it should be in the solution
        self.current_pos = current_pos if current_pos else target_pos.copy()  # Current position
        self.rect = pygame.Rect(self.current_pos[0], self.current_pos[1], image.get_width(), image.get_height())
        self.is_placed_correctly = False
        self.alpha = 255  # Full opacity
        self.is_selected = False
        
        # Create an outline effect for this piece
        self.outline = self.create_outline(image)
    
    def create_outline(self, image, outline_color=(255, 255, 0)):
        """Create an outline effect for the puzzle piece."""
        # Create a mask from the transparent parts of the image
        mask = pygame.mask.from_surface(image)
        outline = mask.outline()
        
        # Create a new transparent surface for the outline
        outline_surface = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        
        # Draw the outline
        if outline:
            pygame.draw.polygon(outline_surface, outline_color, outline, 2)
        
        return outline_surface
    
    def update_position(self, pos):
        self.current_pos = list(pos)
        self.rect.x = pos[0]
        self.rect.y = pos[1]
        
        # Remove snapping behavior - pieces won't snap to target positions
        self.is_placed_correctly = False
        return False
    
    def draw(self, surface):
        # Draw the piece with proper alpha
        if self.alpha < 255:
            # Create a copy with adjusted alpha
            temp = self.image.copy()
            temp.set_alpha(self.alpha)
            surface.blit(temp, self.current_pos)
        else:
            surface.blit(self.image, self.current_pos)
        
        # Draw outline if selected or correctly placed
        if self.is_selected:
            # Draw a yellow outline around the selected piece
            surface.blit(self.outline, self.current_pos)
        elif self.is_placed_correctly:
            # Draw a green border for correctly placed pieces
            pygame.draw.rect(surface, SOLVED_BORDER_COLOR, self.rect, 2)

def convert_svg_to_png(svg_path):
    """Convert an SVG file to a PNG file using a temporary file."""
    # Check if Inkscape is available to convert SVG to PNG
    try:
        output_path = tempfile.mktemp(suffix='.png')
        
        # Try using Inkscape for conversion (if installed)
        try:
            subprocess.run(['inkscape', '--export-filename=' + output_path, svg_path], 
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            # If Inkscape fails or isn't available, try using ImageMagick
            try:
                subprocess.run(['convert', svg_path, output_path], 
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except:
                # If both fail, return a dummy surface as fallback
                print(f"Warning: Could not convert SVG to PNG. Using placeholder for {svg_path}")
                return create_placeholder_image(150, 150)
                
        # Load the PNG into a Pygame surface
        try:
            surface = pygame.image.load(output_path).convert_alpha()
            os.unlink(output_path)  # Remove temporary file
            return surface
        except pygame.error:
            print(f"Error loading converted PNG from {output_path}")
            return create_placeholder_image(150, 150)
            
    except Exception as e:
        print(f"Error during SVG conversion: {e}")
        return create_placeholder_image(150, 150)

def create_placeholder_image(width, height):
    """Create a placeholder image when SVG conversion fails."""
    # Create a surface
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    # Fill with a semi-transparent color
    surface.fill((200, 200, 200, 128))
    # Draw a border
    pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(0, 0, width, height), 2)
    # Draw an X
    pygame.draw.line(surface, (255, 0, 0), (0, 0), (width, height), 2)
    pygame.draw.line(surface, (255, 0, 0), (0, height), (width, 0), 2)
    return surface

def preprocess_svg_directory(directory, output_directory):
    """Pre-process all SVGs in a directory to PNGs for faster loading."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Process all SVG files
    svg_files = [f for f in os.listdir(directory) if f.endswith('.svg')]
    
    print(f"Pre-processing {len(svg_files)} SVG files...")
    
    for i, svg_file in enumerate(svg_files):
        svg_path = os.path.join(directory, svg_file)
        png_path = os.path.join(output_directory, os.path.splitext(svg_file)[0] + '.png')
        
        # Skip if PNG already exists
        if os.path.exists(png_path):
            continue
        
        print(f"Converting {svg_file} to PNG... ({i+1}/{len(svg_files)})")
        
        # Try different conversion methods in order of preference
        try:
            # Method 1: Try Inkscape for best quality conversion
            try:
                subprocess.run([
                    'inkscape',
                    '--export-filename=' + png_path,
                    '--export-background-opacity=0',  # Ensure transparent background
                    svg_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                continue
            except:
                pass
                
            # Method 2: Try ImageMagick
            try:
                subprocess.run([
                    'convert',
                    '-background', 'none',  # Transparent background
                    svg_path,
                    png_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                continue
            except:
                pass
            
            # Method 3: Use PIL directly
            try:
                from PIL import Image
                from cairosvg import svg2png
                
                # Convert SVG to PNG using CairoSVG
                svg2png(url=svg_path, write_to=png_path, background_color=(255, 255, 255, 0))
                continue
            except:
                pass
            
            # If all methods fail, create a placeholder
            print(f"Warning: Could not convert {svg_file} to PNG. Using placeholder.")
            placeholder = create_placeholder_image(150, 150)
            pygame.image.save(placeholder, png_path)
            
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")
            placeholder = create_placeholder_image(150, 150)
            pygame.image.save(placeholder, png_path)
    
    print("Pre-processing complete!")

def extract_puzzle_piece_shape(svg_path, output_path):
    """Simple redirect to conversion methods"""
    try:
        # Try Inkscape
        try:
            subprocess.run([
                'inkscape',
                '--export-filename=' + output_path,
                '--export-background-opacity=0',
                svg_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            pass
        
        # Try ImageMagick
        try:
            subprocess.run([
                'convert',
                '-background', 'none',
                svg_path,
                output_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            pass
        
        # Try CairoSVG
        try:
            from cairosvg import svg2png
            svg2png(url=svg_path, write_to=output_path, background_color=(255, 255, 255, 0))
            return True
        except:
            pass
            
        return False
    except Exception as e:
        print(f"Error in extract_puzzle_piece_shape: {e}")
        return False

def load_puzzle_pieces(svg_directory, cache_directory, rows=5, cols=8, scale_factor=0.15):
    """Load all puzzle pieces from the cache directory or convert from SVGs if needed."""
    global pieces
    pieces = []
    
    # Ensure cache directory exists
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    
    # Pre-process SVGs to PNGs if needed
    preprocess_svg_directory(svg_directory, cache_directory)
    
    # Load the PNG files from cache
    png_files = [f for f in os.listdir(cache_directory) if f.endswith('.png')]
    
    # Sort PNG files by their numeric names
    png_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    if not png_files:
        print("No puzzle pieces found! Make sure you've generated SVGs first.")
        return rows, cols, 150, 150  # Default dimensions
    
    # Load the first image to get dimensions
    first_image = pygame.image.load(os.path.join(cache_directory, png_files[0])).convert_alpha()
    original_width = first_image.get_width()
    original_height = first_image.get_height()
    
    # Scale the pieces down
    piece_width = int(original_width * scale_factor)
    piece_height = int(original_height * scale_factor)
    
    # Calculate grid positioning for the solved puzzle
    grid_start_x = (SCREEN_WIDTH - (cols * piece_width)) // 2
    grid_start_y = (SCREEN_HEIGHT - (rows * piece_height)) // 2
    
    # Define safe margins to ensure pieces are fully visible
    margin_x = piece_width // 2
    margin_y = piece_height // 2
    
    # Calculate usable area for random placement
    usable_width = SCREEN_WIDTH - 2 * margin_x
    usable_height = SCREEN_HEIGHT - 2 * margin_y
    
    # Define regions to avoid placing pieces too close together
    occupied_regions = []
    region_size = max(piece_width, piece_height) + 5  # Add small buffer
    
    # Create puzzle pieces with proper target positions
    for i, png_file in enumerate(png_files):
        # Load the PNG as a Pygame surface
        png_path = os.path.join(cache_directory, png_file)
        
        # Handle transparency better with color key for black backgrounds
        original_surface = pygame.image.load(png_path).convert_alpha()
        
        # Set color key to remove black background
        original_surface.set_colorkey((0, 0, 0))
        
        # Scale the surface
        surface = pygame.transform.smoothscale(original_surface, (piece_width, piece_height))
        
        # Ensure transparency is preserved
        surface.set_colorkey((0, 0, 0))
        if surface.get_alpha() is None:
            surface = surface.convert_alpha()
        
        # Calculate grid position (target position)
        grid_x = i % cols
        grid_y = i // cols
        target_x = grid_start_x + (grid_x * piece_width)
        target_y = grid_start_y + (grid_y * piece_height)
        
        # Starting position range
        start_x_range = (BOX_X + BOX_WIDTH + 50, SCREEN_WIDTH - piece_size - 50)
        start_y_range = (50, SCREEN_HEIGHT - piece_size - 50)
        
        grid_x_offset = BOX_X + 50
        grid_y_offset = BOX_Y + 50
        
        # Random position
        rand_x = random.randint(50, SCREEN_WIDTH - 150)
        rand_y = random.randint(50, SCREEN_HEIGHT - 150)
        
        # Try to avoid overlaps with basic collision check
        attempts = 0
        while attempts < 20:
            rand_x = random.randint(50, SCREEN_WIDTH - 150)
            rand_y = random.randint(50, SCREEN_HEIGHT - 150)
            
            position_valid = True
            new_region = (rand_x, rand_y)
            
            for region in occupied_regions:
                if (abs(region[0] - new_region[0]) < region_size and 
                    abs(region[1] - new_region[1]) < region_size):
                    position_valid = False
                    break
            
            if position_valid or attempts > 15:
                occupied_regions.append(new_region)
                break
            
            attempts += 1
        
        # Create piece
        piece = PuzzlePiece(
            image=surface,
            original_index=int(os.path.splitext(png_file)[0]),
            target_pos=[target_x, target_y],
            current_pos=[rand_x, rand_y]
        )
        pieces.append(piece)
    
    return rows, cols, piece_width, piece_height

def draw_grid(surface, rows, cols, piece_width, piece_height):
    """Draw the solution grid."""
    if not show_grid:
        return
    
    grid_start_x = (SCREEN_WIDTH - (cols * piece_width)) // 2
    grid_start_y = (SCREEN_HEIGHT - (rows * piece_height)) // 2
    
    # Draw grid lines
    for i in range(rows + 1):
        y = grid_start_y + (i * piece_height)
        pygame.draw.line(surface, GRID_COLOR, (grid_start_x, y), 
                         (grid_start_x + cols * piece_width, y), 1)
    
    for i in range(cols + 1):
        x = grid_start_x + (i * piece_width)
        pygame.draw.line(surface, GRID_COLOR, (x, grid_start_y), 
                         (x, grid_start_y + rows * piece_height), 1)

def check_game_complete():
    """Check if all puzzle pieces are correctly placed."""
    # Since pieces no longer snap into place, game is never automatically completed
    return False

def main():
    global selected_piece, dragging, game_complete, show_grid, show_solution_overlay, pieces
    
    # Game setup
    clock = pygame.time.Clock()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces_directory = os.path.join(project_root, "pieces")  # Use pieces folder directly
    
    print(f"Loading puzzle pieces from: {pieces_directory}")
    
    # Create loading screen
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont('Arial', 24)
    loading_text = font.render('Loading puzzle pieces... This may take a moment.', True, (0, 0, 0))
    loading_rect = loading_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
    screen.blit(loading_text, loading_rect)
    pygame.display.flip()
    
    # Load PNG files directly from pieces folder
    png_files = [f for f in os.listdir(pieces_directory) if f.endswith('.png') and os.path.getsize(os.path.join(pieces_directory, f)) > 1000]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not png_files:
        print("No puzzle pieces found in pieces directory!")
        return
        
    # Load the first image to get dimensions
    first_image_path = os.path.join(pieces_directory, png_files[0])
    first_image = pygame.image.load(first_image_path).convert_alpha()
    
    # Scale factor for pieces - make them smaller
    scale_factor = 0.03  # Reduced from 0.06 to make pieces even smaller
    piece_width = int(first_image.get_width() * scale_factor)
    piece_height = int(first_image.get_height() * scale_factor)
    
    # Grid layout
    rows = 5
    cols = 6  # Adjust based on number of pieces
    
    # Calculate grid positioning for the solved puzzle
    grid_start_x = (SCREEN_WIDTH - (cols * piece_width)) // 2
    grid_start_y = (SCREEN_HEIGHT - (rows * piece_height)) // 2
    
    # Define safe margins to ensure pieces are fully visible
    margin_x = piece_width // 2
    margin_y = piece_height // 2
    
    # Calculate usable area for random placement
    usable_width = SCREEN_WIDTH - 2 * margin_x - piece_width
    usable_height = SCREEN_HEIGHT - 2 * margin_y - piece_height
    
    # For avoiding placing pieces too close together
    occupied_regions = []
    region_size = max(piece_width, piece_height) + 5  # Add small buffer
    
    # Clear existing pieces
    pieces = []
    
    # Load and create pieces
    for i, png_file in enumerate(png_files):
        # Load the PNG as a Pygame surface
        png_path = os.path.join(pieces_directory, png_file)
        
        try:
            # Handle transparency better with color key for black backgrounds
            original_surface = pygame.image.load(png_path).convert_alpha()
            
            # Set color key to remove black background
            original_surface.set_colorkey((0, 0, 0))
            
            # Scale the surface
            surface = pygame.transform.smoothscale(original_surface, (piece_width, piece_height))
            
            # Ensure transparency is preserved
            surface.set_colorkey((0, 0, 0))
            if surface.get_alpha() is None:
                surface = surface.convert_alpha()
            
            # Calculate grid position (target position)
            grid_x = i % cols
            grid_y = i // cols
            target_x = grid_start_x + (grid_x * piece_width)
            target_y = grid_start_y + (grid_y * piece_height)
            
            # Random position
            rand_x = random.randint(50, SCREEN_WIDTH - 150)
            rand_y = random.randint(50, SCREEN_HEIGHT - 150)
            
            # Try to avoid overlaps with basic collision check
            attempts = 0
            while attempts < 20:
                rand_x = random.randint(50, SCREEN_WIDTH - 150)
                rand_y = random.randint(50, SCREEN_HEIGHT - 150)
                
                position_valid = True
                new_region = (rand_x, rand_y)
                
                for region in occupied_regions:
                    if (abs(region[0] - new_region[0]) < region_size and 
                        abs(region[1] - new_region[1]) < region_size):
                        position_valid = False
                        break
                
                if position_valid or attempts > 15:
                    occupied_regions.append(new_region)
                    break
                
                attempts += 1
            
            # Get piece number from filename
            try:
                piece_num = int(png_file.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                piece_num = i + 1
            
            # Create piece
            piece = PuzzlePiece(
                image=surface,
                original_index=piece_num,
                target_pos=[target_x, target_y],
                current_pos=[rand_x, rand_y]
            )
            pieces.append(piece)
            print(f"Loaded piece {piece_num}")
            
        except Exception as e:
            print(f"Error loading {png_file}: {e}")
    
    print(f"Successfully loaded {len(pieces)} pieces")
    
    # Main game loop
    running = True
    offset_x, offset_y = 0, 0  # Initialize offset variables
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Toggle grid with G key
                    if event.key == pygame.K_g:
                        show_grid = not show_grid
                    
                    # Show solution overlay with S key
                    elif event.key == pygame.K_s:
                        show_solution_overlay = not show_solution_overlay
                        
                        # Adjust transparency of pieces
                        for piece in pieces:
                            if show_solution_overlay:
                                piece.alpha = 128
                            else:
                                piece.alpha = 255
                    
                    # Reset puzzle with R key
                    elif event.key == pygame.K_r:
                        # Randomize positions of all pieces
                        for piece in pieces:
                            rand_x = random.randint(50, SCREEN_WIDTH - piece_width - 50)
                            rand_y = random.randint(50, SCREEN_HEIGHT - piece_height - 50)
                            piece.update_position([rand_x, rand_y])
                            piece.is_placed_correctly = False
                        game_complete = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        # Reset selection
                        if selected_piece:
                            selected_piece.is_selected = False
                            
                        # Check if we clicked on a piece
                        for piece in reversed(pieces):  # Reverse to check top pieces first
                            # Check for click using mask collision for precise shape detection
                            pos_in_piece = (event.pos[0] - piece.rect.x, event.pos[1] - piece.rect.y)
                            if piece.rect.collidepoint(event.pos) and 0 <= pos_in_piece[0] < piece.image.get_width() and 0 <= pos_in_piece[1] < piece.image.get_height():
                                # Check if the pixel at this position has alpha > 0
                                try:
                                    if piece.image.get_at(pos_in_piece)[3] > 0:  # Alpha > 0 means not transparent
                                        selected_piece = piece
                                        piece.is_selected = True
                                        # Move the selected piece to the end of the list (top of the pile)
                                        pieces.remove(piece)
                                        pieces.append(piece)
                                        dragging = True
                                        # Store offset for smooth dragging
                                        offset_x = piece.rect.x - event.pos[0]
                                        offset_y = piece.rect.y - event.pos[1]
                                        break
                                except IndexError:
                                    # If we get an index error, the pixel is outside the image's actual dimensions
                                    pass
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and dragging:  # Left mouse button release
                        dragging = False
                        if selected_piece:
                            # No longer check if piece is near its target position
                            selected_piece.is_selected = False
                            selected_piece = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging and selected_piece:
                        # Update piece position
                        mouse_x, mouse_y = event.pos
                        selected_piece.update_position((mouse_x + offset_x, mouse_y + offset_y))
            
            # Draw everything
            screen.fill(BG_COLOR)
            
            # Draw grid
            draw_grid(screen, rows, cols, piece_width, piece_height)
            
            # Draw solution overlay if enabled
            if show_solution_overlay:
                grid_start_x = (SCREEN_WIDTH - (cols * piece_width)) // 2
                grid_start_y = (SCREEN_HEIGHT - (rows * piece_height)) // 2
                
                for i, piece in enumerate(sorted(pieces, key=lambda p: p.original_index)):
                    grid_x = i % cols
                    grid_y = i // cols
                    target_x = grid_start_x + (grid_x * piece_width)
                    target_y = grid_start_y + (grid_y * piece_height)
                    
                    # Draw a semi-transparent version at the target position
                    temp = piece.image.copy()
                    temp.set_alpha(100)  # Very transparent
                    screen.blit(temp, (target_x, target_y))
            
            # Draw pieces
            for piece in pieces:
                piece.draw(screen)
            
            # Display game complete message
            if game_complete:
                font = pygame.font.SysFont('Arial', 36)
                text = font.render('Puzzle Complete! Press R to restart', True, (0, 200, 0))
                text_rect = text.get_rect(center=(SCREEN_WIDTH//2, 50))
                screen.blit(text, text_rect)
            
            # Draw controls info
            font = pygame.font.SysFont('Arial', 16)
            controls = [
                "Click and drag pieces to move them freely",
                "S: Show/Hide Solution Overlay",
                "R: Reset puzzle positions",
                "ESC: Quit game"
            ]
            for i, control in enumerate(controls):
                text = font.render(control, True, (50, 50, 50))
                screen.blit(text, (20, SCREEN_HEIGHT - 100 + i*20))
            
            pygame.display.flip()
            clock.tick(60)
            
    except Exception as e:
        print(f"Error in game loop: {e}")
        
    finally:
        # Clean up resources
        print("Shutting down...")
        # Force immediate exit to avoid hanging
        os._exit(0)

if __name__ == "__main__":
    main() 