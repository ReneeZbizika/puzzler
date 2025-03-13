import os
import json
from PIL import Image, ImageDraw

def main():
    # Load the centroid data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "Datasets", "puzzle_centroids.json")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Create a blank overlay image
    overlay = Image.new("RGBA", (1200, 800), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw the puzzle box
    BOX_X, BOX_Y = 50, 50
    box_width, box_height = 516.8, 372.4  # From your calculations
    draw.rectangle((BOX_X, BOX_Y, BOX_X + box_width, BOX_Y + box_height), 
                   outline=(180, 180, 180, 128), width=2)
    
    # Draw a grid
    grid_width, grid_height = 5, 5
    cell_width = box_width / grid_width
    cell_height = box_height / grid_height
    
    # Draw vertical grid lines
    for i in range(1, grid_width):
        x = BOX_X + i * cell_width
        draw.line((x, BOX_Y, x, BOX_Y + box_height), fill=(200, 200, 200, 64), width=1)
    
    # Draw horizontal grid lines
    for i in range(1, grid_height):
        y = BOX_Y + i * cell_height
        draw.line((BOX_X, y, BOX_X + box_width, y), fill=(200, 200, 200, 64), width=1)
    
    # Draw each centroid as a colored dot
    for i, piece_data in enumerate(data["pieces"]):
        centroid = piece_data["centroid"]
        x, y = centroid["x"], centroid["y"]
        
        # Alternating colors for better visibility
        color = (255, 0, 0, 192) if i % 2 == 0 else (0, 0, 255, 192)
        
        # Draw a circle at centroid position
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
        
        # Draw piece ID
        piece_id = piece_data["id"].split("_")[1]  # Extract just the number
        draw.text((x+8, y-8), piece_id, fill=(0, 0, 0, 192))
    
    # Save the overlay
    overlay_path = os.path.join(project_root, "data", "centroid_verification.png")
    overlay.save(overlay_path)
    print(f"Verification image saved to {overlay_path}")
    
    # Try to open the image
    try:
        overlay.show()
    except Exception as e:
        print(f"Could not display image: {e}")
        print(f"Image saved to {overlay_path}")

if __name__ == "__main__":
    main() 