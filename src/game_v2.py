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

def check_game_complete():
    """Check if all puzzle pieces are correctly placed."""
    return all(piece.placed_correctly for piece in pieces)

def main():
    global selected_piece, dragging, game_complete
    
    # Generate puzzle pieces
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

if __name__ == "__main__":
    main()
