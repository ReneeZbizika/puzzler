import os
import json

def load_json(filename):
    """Load JSON file and return the data."""
    with open(filename, "r") as f:
        return json.load(f)

def generate_answer_key(param_file_path):
    """
    Generate an answer key dynamically based on parameters from the JSON file.
    """
    # Load parameters from JSON
    data = load_json(param_file_path)
    
    BOX_X, BOX_Y = data["BOX_WIDTH"], data["BOX_HEIGHT"]
    rows, cols = data["Num Row Pieces"], data["Num Col Pieces"]
    cell_width = data["Scaled dimensions"][0] / cols
    cell_height = data["Scaled dimensions"][1] / rows

    answer_key = {
        piece_id: {
            "x": BOX_X + (c * cell_width),
            "y": BOX_Y + (r * cell_height)
        }
        for r in range(rows)
        for c in range(cols)
        for piece_id in [r * cols + c + 1]  # Generate unique piece IDs dynamically
    }

    return answer_key

def save_answer_key(param_file_path, output_filename="Datasets/evaluation/answer_key/answer_key.json"):
    """Generate and save the answer key based on the provided JSON configuration."""
    answer_key = generate_answer_key(param_file_path)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(answer_key, f, indent=4)
    print(f"Answer key saved to {output_filename}")

if __name__ == "__main__":
    param_file_path = "path/to/your/parameters.json"  # Update this with the correct path
    save_answer_key(param_file_path)
