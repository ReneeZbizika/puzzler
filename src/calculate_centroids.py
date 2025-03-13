import os
import json
import pygame
import numpy as np
from PIL import Image

# Initialize pygame
pygame.init()

# Constants from game_v2.py
BOX_X = 50
BOX_Y = 50

def calculate_puzzle_layout(piece_count=25):
    """
    Calculate a layout grid for the puzzle pieces based on the total count.
    Approximates a layout for a Thomas Kinkade cottage scene (5x5 puzzle)
    """
    # For the winter cottage puzzle, we'll use a 5x5 grid as shown in the image
    grid_width = 5
    grid_height = 5
    
    # Adjust if different piece count
    if piece_count != 25:
        grid_width = int(np.ceil(np.sqrt(piece_count)))
        grid_height = int(np.ceil(piece_count / grid_width))
    
    return grid_width, grid_height

def get_puzzle_dimensions():
    """Get the puzzle box dimensions from the parameters file"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(project_root, "params", "img_2_params.txt")
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                if line.startswith("Scaled dimensions:"):
                    dims = line.strip().split(': ')[1]
                    width, height = dims.split('x')
                    return int(width) * 3.8, int(height) * 3.8
    except FileNotFoundError:
        print(f"Params file not found: {param_file}")
        print("Using default dimensions")
    
    # Default values if unable to read from file - based on the cottage puzzle image
    return 600, 400

def calculate_centroids():
    """Calculate the centroids for each puzzle piece in the assembled state"""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pieces_path = os.path.join(project_root, "pieces_img_2")
    
    # Check if the folder exists
    if not os.path.exists(pieces_path):
        print(f"Error: Pieces folder '{pieces_path}' not found!")
        return []
    
    # Get all PNG files in the folder
    piece_files = [f for f in os.listdir(pieces_path) if f.endswith('.png')]
    if not piece_files:
        print(f"Error: No PNG files found in '{pieces_path}'")
        return []
    
    # Sort the files by their piece number
    piece_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"Processing {len(piece_files)} puzzle pieces")
    
    # Get puzzle box dimensions - for the cottage puzzle
    box_width, box_height = get_puzzle_dimensions()
    print(f"Puzzle box dimensions: {box_width}x{box_height}")
    
    # Calculate grid layout
    grid_width, grid_height = calculate_puzzle_layout(len(piece_files))
    print(f"Using grid layout: {grid_width}x{grid_height}")
    
    # Calculate cell size
    cell_width = box_width / grid_width
    cell_height = box_height / grid_height
    print(f"Cell size: {cell_width}x{cell_height}")
    
    # List to store calculated centroids
    centroids = []
    
    # Process each piece
    for i, piece_file in enumerate(piece_files):
        piece_id = os.path.splitext(piece_file)[0]  # Get filename without extension
        
        # Load the piece image to get its dimensions
        piece_path = os.path.join(pieces_path, piece_file)
        try:
            # Use PIL instead of pygame
            piece_image = Image.open(piece_path)
            img_width, img_height = piece_image.size
            
            # Calculate grid position (row, col)
            row = i // grid_width
            col = i % grid_width
            
            # Calculate center position within the puzzle box
            center_x = BOX_X + (col * cell_width) + (cell_width / 2)
            center_y = BOX_Y + (row * cell_height) + (cell_height / 2)
            
            centroids.append({
                "id": piece_id,
                "centroid": {
                    "x": round(center_x),
                    "y": round(center_y)
                }
            })
            
            print(f"Piece {piece_id}: calculated centroid at ({round(center_x)}, {round(center_y)})")
            
        except Exception as e:
            print(f"Error processing piece {piece_file}: {e}")
    
    return centroids

def update_json_file():
    """Calculate centroids and update the JSON file"""
    centroids = calculate_centroids()
    
    if not centroids:
        print("No centroids calculated. Check for errors above.")
        return
    
    # Create data structure for JSON
    data = {"pieces": centroids}
    
    # Write to JSON file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "data", "puzzle_centroids.json")
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully wrote {len(centroids)} centroid coordinates to {json_path}")

if __name__ == "__main__":
    update_json_file() 