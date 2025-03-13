import os
import json

def generate_answer_key():
    """
    Generate an answer key for a puzzle arranged in a grid.
    
    For example, suppose the solution box's top-left is at (BOX_X, BOX_Y)
    and each cell in the grid has fixed dimensions.
    Adjust these parameters as needed.
    """
    # Define solution box top-left and cell size (in pixels)
    BOX_X = 100
    BOX_Y = 100
    cell_width = 50
    cell_height = 50
    
    # Define grid dimensions (rows x cols)
    rows, cols = 5, 5
    
    answer_key = {}
    piece_id = 1
    for r in range(rows):
        for c in range(cols):
            target_x = BOX_X + c * cell_width
            target_y = BOX_Y + r * cell_height
            # Save as dictionary entries; you can change the format as needed.
            answer_key[piece_id] = {"x": target_x, "y": target_y}
            piece_id += 1
    return answer_key

def save_answer_key(filename="Datasets/evaluation/answer_key/answer_key.json"):
    answer_key = generate_answer_key()
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(answer_key, f, indent=4)
    print(f"Answer key saved to {filename}")

if __name__ == "__main__":
    save_answer_key()
