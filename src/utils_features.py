# Edge Compatibility Algorithm 
# Inspired by https://www-users.cse.umn.edu/~olver/v_/puzzles.pdf

import numpy as np
import cv2
import math
import scipy as sp
import itertools
import os
import json
import pygame
import cv2
from skimage.metrics import structural_similarity as ssim

from env import img_name, image_name, root_eval
# env variables
root = "Datasets"
from env import BOX_WIDTH, BOX_HEIGHT, BOX_X, BOX_Y, SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR

#TODO: write tests
#TODO: check if the numbers generated in edge compatibility matrix are accurate 
#TODO: switch off of segment_piece_black_bg to segment_piece

# ---------- Edge Compatibility Helper ----------

def load_image(path): 
    """Load an image in grayscale.""" 
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    return image

def segment_piece(image): 
    """ Segment the puzzle piece from the background. 
    This example uses a simple threshold and finds the largest external contour. """ 
    
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV) 
    #ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print("Threshold return value:", ret)
    #cv2.imwrite("thresh_debug.png", thresh)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    if not contours:
        raise ValueError("No contours found. Adjust thresholding or check the image quality.")

    # Choose the largest contour by area (assuming one piece per image) 
    largest_contour = max(contours, key=cv2.contourArea) 
    # Remove redundant dimensions; resulting shape is (n, 2) 
    contour_points = largest_contour.squeeze() 
    return contour_points

def segment_piece_black_bg(image_path):
    # 1. Load the image in color first to check
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 2. Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    #print("Min pixel value:", np.min(color_img))
    #print("Max pixel value:", np.max(color_img))
    #print("Mean pixel value:", np.mean(color_img))
    #print(color_img.shape)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #print("Loaded image shape:", image.shape)

    # 3. Threshold for near-black background
    #    If background is near pure black, a low threshold like 10 or 15 can help.
    #    'THRESH_BINARY' will produce a white piece on black background.
    #    Adjust the threshold value to suit your images.
    _, thresh = cv2.threshold(gray_img, 5, 255, cv2.THRESH_BINARY)
    
    # 4. Find external contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Check your threshold or image quality.")
    
    # 5. Pick the largest contour (assuming a single piece in the image)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze()

    return contour_points, thresh

def compute_signature(contour): 
    """ Compute a discrete signature for a contour. 
    For each point on the contour, approximate the curvature and its derivative.
    This is a simple finite-difference method and may be refined. """ 
    n = contour.shape[0] 
    curvature = np.zeros(n)
    # Compute curvature at each point using three consecutive points.
    for i in range(n):
        prev = contour[i - 1]
        curr = contour[i]
        next = contour[(i + 1) % n]
        
        # Compute vectors from current point to previous and next points.
        v1 = prev - curr
        v2 = next - curr
        
        # Compute the angle between v1 and v2.
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        # Normalize the angle between -pi and pi.
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        
        # Approximate curvature as the angle change divided by average arc length.
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        curvature[i] = angle / ((l1 + l2) / 2 + 1e-5)

    # Estimate the derivative of curvature with respect to arc length.
    # First, compute cumulative arc-length.
    distances = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    arc_length = np.concatenate(([0], np.cumsum(distances)))
    # Use numpy's gradient to approximate derivative.
    curvature_derivative = np.gradient(curvature, arc_length)

    # Stack curvature and its derivative to form the signature.
    signature = np.stack((curvature, curvature_derivative), axis=1)
    return signature

# Compute a discrete "signature" that approximate the curvature and its derivative along the edge
def resample_edge(signature, num_points): 
    """
    Resample the given edge signature to num_points using interpolation. 
    
    Params:
    signature (np.array): An array of shape (n, 2) with (kappa, kappa_s) values.
    num_points (int): The desired number of samples.

    Returns:
    np.array: An array of shape (num_points, 2) with the resampled signature.
    """
    n = signature.shape[0]
    # Parameterize the original signature uniformly along [0,1)
    t_original = np.linspace(0, 1, n, endpoint=False)
    t_new = np.linspace(0, 1, num_points, endpoint=False)

    # Create an interpolation function (using cubic interpolation here)
    interpolator = sp.interpolate.interp1d(t_original, signature, axis=0, kind='cubic', fill_value="extrapolate")
    resampled_signature = interpolator(t_new)
    return resampled_signature

# Define an edge compatibility score as a function of how "close" these signature vectors are
# Compatibility score between 2 edges
# greater alpha = stricter 
def edge_compatibility(edge1_signature, edge2_signature, alpha=1.0): 
    # Ensure both edges are sampled to the same number of points: 
    if edge1_signature.shape[0] != edge2_signature.shape[0]: 
        raise ValueError("Resampled edges must have the same number of points")
    
    num_points = max(len(edge1_signature), len(edge2_signature)) 
    #sig1 = resample_edge(edge1_signature, num_points) 
    #sig2 = resample_edge(edge2_signature, num_points)
    
    # Compute the pointwise Euclidean distance between the signatures:
    # Here, each signature is a 2D point (kappa, kappa_s)
    differences = np.linalg.norm(edge1_signature - edge2_signature, axis=1)
    # Compute the mean difference:
    mean_diff = np.mean(differences)
    # Convert the mean difference to a score between 0 and 1:
    # Here we chose a simple exponential decay: (possible to use sigmoid function?)
    score = np.exp(-alpha * mean_diff)
    
    return score

def segment_piece_black_bg(image_path):
    """Segments a puzzle piece from a black background and extracts its bounding box."""
    
    # Load image
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray_img, 5, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError(f"No contours found in {image_path}. Check your threshold or image quality.")
    
    # Select the largest contour (assuming a single puzzle piece)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute the bounding box (min area rectangle)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # Convert to integer coordinates

    return largest_contour.squeeze(), box  # Ensure only two values are returned

def extract_four_edges(contour, bounding_box):
    # Sort bounding box points to ensure order: top-left, top-right, bottom-right, bottom-left
    box = sorted(bounding_box, key=lambda p: (p[1], p[0]))  # Sort by y, then x
    
    top_left, top_right = sorted(box[:2], key=lambda p: p[0])  # Top two points
    bottom_left, bottom_right = sorted(box[2:], key=lambda p: p[0])  # Bottom two points

    # Define four edge segments based on bounding box
    edges = {
        "top": [],
        "right": [],
        "bottom": [],
        "left": []
    }

    # Loop through the contour and assign each point to the closest edge
    for point in contour:
        x, y = point

        # Compute distances to each of the four edges
        dist_top = abs(y - top_left[1])  # Distance to top edge
        dist_bottom = abs(y - bottom_left[1])  # Distance to bottom edge
        dist_left = abs(x - top_left[0])  # Distance to left edge
        dist_right = abs(x - top_right[0])  # Distance to right edge

        # Assign to the closest edge
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

        if min_dist == dist_top:
            edges["top"].append(point)
        elif min_dist == dist_bottom:
            edges["bottom"].append(point)
        elif min_dist == dist_left:
            edges["left"].append(point)
        else:
            edges["right"].append(point)

    # Convert lists to numpy arrays
    for key in edges.keys():
        edges[key] = np.array(edges[key])

    return [edges["top"], edges["right"], edges["bottom"], edges["left"]]
    
def extract_edges_from_pieces(folder_path, num_resample_points=100):
    """Extracts all four edges from each puzzle piece and returns their signatures."""
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))])
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    
    edge_signatures = []  
    edge_to_piece_map = {}  
    edge_counter = 0

    for piece_id, image_path in enumerate(image_paths):
        try:
            # Extract the puzzle piece contour and bounding box
            contour, bounding_box = segment_piece_black_bg(image_path)
            
            # Extract the four edges from the contour
            edges = extract_four_edges(contour, bounding_box)

            for edge_id, edge in enumerate(edges):
                if edge.shape[0] < 5:  # Avoid very small edges that could cause errors
                    print(f"Skipping small edge {edge_id} for piece {piece_id}")
                    continue

                # Compute signature for each edge
                signature = compute_signature(edge)
                resampled_signature = resample_edge(signature, num_resample_points) if signature.shape[0] > 3 else signature

                edge_signatures.append(resampled_signature)
                edge_name = f"edge_{edge_counter}"
                edge_to_piece_map[edge_name] = (piece_id, edge_id)  # Tracking four edges per piece
                edge_counter += 1
        except ValueError as e:
            print(f"Skipping {image_path}: {e}")

    return edge_signatures, edge_to_piece_map, image_files

#TODO check if edge_to_piece_map returns actually correct mapping
def save_edge_to_piece_map(edge_to_piece_map, file_path="Datasets/evaluation/edge_to_piece_map.json"):
    """
    Save the edge_to_piece map to a JSON file.
    
    Parameters:
      edge_to_piece_map (dict): Mapping from edge name to (piece_id, edge_index) tuple.
      file_path (str): Destination file path.
    """
    # Convert tuple values to lists so that they are JSON serializable.
    serializable_map = {key: list(value) for key, value in edge_to_piece_map.items()}
    with open(file_path, "w") as f:
        json.dump(serializable_map, f, indent=4)
    print(f"Edge-to-piece map saved to {file_path}")

def load_edge_to_piece_map(root_eval, img_name):
    """
    Load the edge_to_piece map from a JSON file.
    
    Returns:
      dict: A mapping from edge name to (piece_id, edge_index) tuple.
    """
    # Validate that both parameters are strings.
    if not isinstance(root_eval, str):
        raise TypeError(f"Expected root_eval to be a string, got {type(root_eval)}")
    if not isinstance(img_name, str):
        print(img_name)
        raise TypeError(f"Expected img_name to be a string, got {type(img_name)}")
    
    # Construct the file path by appending ".json"
    file_path = os.path.join(root_eval, f"{img_name}_edge_to_piece_map.json")
    
    with open(file_path, "r", encoding="utf-8") as f:
        serializable_map = json.load(f)
    # Convert list values back to tuples.
    edge_to_piece_map = {key: tuple(value) for key, value in serializable_map.items()}
    return edge_to_piece_map

def compute_compatibility_matrix(edge_signatures, alpha=1.0):
    """
    Computes compatibility scores between all extracted edges.
    
    Params:
        edge_signatures (list): A list of resampled edge signatures.
        alpha (float): Sensitivity factor for similarity scoring.
    
    Returns:
        np.array: Compatibility matrix where entry (i, j) is the compatibility score between edge i and edge j.
    """
    num_edges = len(edge_signatures)
    compatibility_matrix = np.zeros((num_edges, num_edges))  # Square matrix for edges
    
    # Compute pairwise compatibility scores
    for (i, edge_A), (j, edge_B) in itertools.combinations(enumerate(edge_signatures), 2):
        score = edge_compatibility(edge_A, edge_B, alpha)
        compatibility_matrix[i, j] = score
        compatibility_matrix[j, i] = score  # Symmetric matrix

    return compatibility_matrix


# ---------- Script to Generate Compatibility Matrix ----------
if __name__ == "__main__":
    #root_eval = "Datasets/evaluation" 
    # ^ already defined in globals
    base_name = os.path.splitext(image_name)[0]
    
    # Use the base name to construct folder and file names.
    folder_path = f"pieces_{base_name}"
    
    # Extract edges and compute compatibility
    edge_signatures, edge_to_piece_map, image_files = extract_edges_from_pieces(folder_path)
    save_edge_to_piece_map(edge_to_piece_map, file_path=f"{root_eval}/{image_name}_edge_to_piece_map.json")
    compatibility_matrix = compute_compatibility_matrix(edge_signatures)
    np.savetxt(f"{root_eval}/{base_name}_edge_compatibility_matrix.csv", compatibility_matrix, delimiter=",", fmt="%.4f")

    print("Edge-to-Piece Mapping:", edge_to_piece_map)
    print("Edge Compatibility Matrix:\n", compatibility_matrix)
    

# ---------- Edge Compatibility Helper (using precomputed Compatibility Matrix) ----------
def load_compatibility_matrix(root_eval, img_name):
    csv_file = os.path.join(root_eval, f"{img_name}_edge_compatibility_matrix.csv")
    return np.loadtxt(csv_file, delimiter=',')

# compatibility_matrix = load_compatibility_matrix("Datasets/edge_compatibility.csv")

#TODO: dynamic load for path_to_comp_matrix

# Example usage:
# piece_id_to_search = 3  # Change this to the piece ID you want to search for
#edges_and_scores = get_piece_edges_and_scores(piece_id_to_search, edge_to_piece_map, compatibility_matrix)

# Print results
#for edge_id, data in edges_and_scores.items():
    #print(f"Edge {edge_id} (Edge Index {data['edge_index']}) has compatibility scores:")
    #sorted_scores = sorted(data["scores"].items(), key=lambda x: -x[1])[:5]  # Show top 5 matches
    #for other_edge, score in sorted_scores:
       # print(f"  → Edge {other_edge}: Score {score:.4f}")
    #print()

def are_edges_adjacent(assembly_state, piece_1, edge_idx_1, piece_2, edge_idx_2):
    """
    Checks if two edges belong to adjacent pieces in the current assembly.
    
    Params:
        assembly_state (dict): Maps piece_id to its (x, y, rotation).
        piece_1, piece_2 (int): IDs of two pieces.
        edge_idx_1, edge_idx_2 (int): Edge indices (0=Top, 1=Right, 2=Bottom, 3=Left).

    Returns:
        bool: True if the edges are adjacent.
    """
    if piece_1 not in assembly_state or piece_2 not in assembly_state:
        return False  # One of the pieces is not placed

    x1, y1, rotation1 = assembly_state[piece_1]
    x2, y2, rotation2 = assembly_state[piece_2]

    # Define adjacency rules based on (x, y) positions
    adjacency_rules = {
        (0, 2): (0, -1),  # Top (0) ↔ Bottom (2) → piece_2 is above piece_1
        (1, 3): (1, 0),   # Right (1) ↔ Left (3) → piece_2 is to the right
        (2, 0): (0, 1),   # Bottom (2) ↔ Top (0) → piece_2 is below
        (3, 1): (-1, 0)   # Left (3) ↔ Right (1) → piece_2 is to the left
    }

    if (edge_idx_1, edge_idx_2) in adjacency_rules:
        dx, dy = adjacency_rules[(edge_idx_1, edge_idx_2)]
        return (x2 == x1 + dx) and (y2 == y1 + dy)

    return False


#TODO: dynamic naming for any image
# edge_signatures, edge_to_piece_map, image_files = extract_edges_from_pieces(folder_path)
#file_path=f"Datasets/evaluation/{folder_path}_edge_to_piece_map.json"
#load_edge_to_piece_map(file_path="Datasets/evaluation/{image_name}_edge_to_piece_map.json")
# folder_path = "pieces_img_2"
# image_name = "img_2"

def evaluate_assembly_compatibility(assembly_state, image_name):
    """
    Evaluates the overall edge compatibility of the current puzzle assembly.

    Params:
        assembly_state (dict): Maps piece_id to its placement info (position, orientation, etc.).
        edge_to_piece_map (dict): Maps edge ID to (piece ID, edge index).
        edge_compatibility (np.array): Precomputed compatibility matrix for edges.

    Returns:
        float: Overall compatibility score of the assembled pieces.
    """
    path_to_edge_compatibility = f"{root_eval}/{image_name}_edge_compatibility_matrix.csv"
    
    # Load edge_to_piece_map
    edge_to_piece_map = load_edge_to_piece_map(root_eval, image_name)
    
    # Load precomputed compat matrix
    comp_matrix = load_compatibility_matrix(root_eval, image_name)
    
    total_score = 0.0
    num_connections = 0

    for edge_id_1, (piece_1, edge_idx_1) in edge_to_piece_map.items():
        for edge_id_2, (piece_2, edge_idx_2) in edge_to_piece_map.items():
            if piece_1 != piece_2:  # Avoid self-matching
                if are_edges_adjacent(assembly_state, piece_1, edge_idx_1, piece_2, edge_idx_2):
                    # Get compatibility score from matrix
                    score = comp_matrix[edge_id_1, edge_id_2]
                    total_score += score
                    num_connections += 1

    return total_score / max(num_connections, 1)  # Avoid division by zero

# ---------- Image Similarity Helper ----------

import torch
import numpy as np

def load_eval_image(root_eval, img_name):
    """
    Load the evaluation image from the given root directory and image name.
    
    Parameters:
        root_eval (str): The root directory where the evaluation images are stored.
        img_name (str): The name of the evaluation image file (e.g., "img_2_evaluate.png").
    
    Returns:
        np.array: The loaded image in RGB format.
        
    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    file_path = os.path.join(root_eval, f"{image_name}_evaluate.png")
    target_img = cv2.imread(file_path)
    if target_img is None:
        raise FileNotFoundError(f"Could not load image from {file_path}")
    
    # Convert the image from BGR (OpenCV default) to RGB.
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    return target_img

#TODO move pygame function to a differet file?
#TODO assume render in game_agent, MCTS, Trainer, and Train is ON by default
# only box is true by default
def render_state_to_image(state, use_screen=True, only_box=True):
    """
    Render the current puzzle state to an image array.
    If use_screen is True and a Pygame display surface exists, capture the current screen.
    Otherwise, render the state off-screen to a new surface.
    
    Parameters:
        state: The current puzzle state.
        use_screen (bool): Whether to capture the current display surface.
        only_box (bool): If True, return only the region inside the grey solution box.
        
    Returns:
        A NumPy array representing the rendered area (in RGB).
    """
    if use_screen:
        screen = pygame.display.get_surface()
        if screen is not None:
            # Capture the entire screen.
            img_array = pygame.surfarray.array3d(screen)
            img_array = np.transpose(img_array, (1, 0, 2))
            if only_box:
                img_array = img_array[BOX_Y:BOX_Y+BOX_HEIGHT, BOX_X:BOX_X+BOX_WIDTH, :]
            return img_array

    # Off-screen rendering:
    offscreen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    offscreen.fill(BG_COLOR)
    pygame.draw.rect(offscreen, BG_COLOR, (BOX_X, BOX_Y, BOX_WIDTH, BOX_HEIGHT))
    for piece in state.pieces.values():
        piece.draw(offscreen)
    
    # Convert to NumPy array.
    img_array = pygame.surfarray.array3d(offscreen)
    img_array = np.transpose(img_array, (1, 0, 2))
    
    if only_box:
        # Crop to the solution box region.
        img_array = img_array[BOX_Y:BOX_Y+BOX_HEIGHT, BOX_X:BOX_X+BOX_WIDTH, :]
    
    return img_array
        

def resize_to_match(img1, img2):
    """Resize img1 to match the dimensions of img2"""
    # Ensure both are numpy arrays
    if not isinstance(img1, np.ndarray):
        print(f"Error: img1 is not a numpy array, it's {type(img1)}")
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    if not isinstance(img2, np.ndarray):
        print(f"Error: img2 is not a numpy array, it's {type(img2)}")
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Check if images are valid
    if img1.size == 0 or img2.size == 0:
        print("Error: One of the images is empty")
        return np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Get dimensions
    h2, w2 = img2.shape[:2]
    
    # Ensure dimensions are positive
    if h2 <= 0 or w2 <= 0:
        print(f"Error: Invalid target dimensions: {w2}x{h2}")
        h2, w2 = max(1, h2), max(1, w2)
    
    # Resize
    resized_img1 = cv2.resize(img1, (w2, h2))
    return resized_img1, img2

#TODO: later implement a choice of metrics between MSE vs SSIM
# SSIM is considered more perceptually relevant than metrics like Mean Squared Error (MSE)
def compute_similarity_score(current_img, target_img):
    """
    Compute a similarity score between the current assembly image and the target image using SSIM.
    
    SSIM returns a value between -1 and 1 (where 1 is perfect similarity). 
    For our purposes, we return the SSIM score directly.
    
    Parameters:
        current_img (np.array): The current state image (RGB).
        target_img (np.array): The target evaluation image (RGB).
    
    Returns:
        float: The SSIM similarity score.
    """
    # Ensure that both images are numpy arrays.
    current_img = np.asarray(current_img)
    target_img = np.asarray(target_img)
    
    # Resize if needed
    current_img, target_img = resize_to_match(current_img, target_img)
    
    # Convert current_img to uint8 if it is not already.
    if current_img.dtype != np.uint8:
        # Normalize if necessary.
        if current_img.max() <= 1.0:
            current_img = (current_img * 255).astype(np.uint8)
        else:
            current_img = current_img.astype(np.uint8)

    # Convert target_img to uint8 if it is not already.
    if target_img.dtype != np.uint8:
        if target_img.max() <= 1.0:
            target_img = (target_img * 255).astype(np.uint8)
        else:
            target_img = target_img.astype(np.uint8)
            
    # Compute SSIM with multichannel=True so that the comparison is done on all color channels.
    # Determine an appropriate win_size.
    # SSIM requires win_size to be an odd value less than or equal to the smallest dimension.
    min_side = min(current_img.shape[0], current_img.shape[1])
    if min_side < 7:
        # If the image is smaller than 7 pixels, use the largest odd number not exceeding min_side.
        win_size = min_side if min_side % 2 == 1 else min_side - 1
    else:
        win_size = 7
        
    score, _ = ssim(current_img, target_img, win_size=win_size, channel_axis=-1, full=True)
    return score

# load target image
# cur state = render_state_to_image
# compute (cur, target)
def extract_visual_features(state, image_name):
    """Extract visual features from the current state"""
    # Handle custom State object
    if hasattr(state, '__class__') and state.__class__.__name__ == 'State':
        # Extract the visual representation from your custom State object
        # This depends on how your State class is structured
        if hasattr(state, 'board') and hasattr(state.board, 'render'):
            # If your State has a board with a render method
            current_img = state.board.render()
        elif hasattr(state, 'render'):
            # If your State has a direct render method
            current_img = state.render()
        elif hasattr(state, 'image'):
            # If your State stores an image directly
            current_img = state.image
        elif hasattr(state, 'pieces'):
            # If your State has puzzle pieces, render them to an image
            current_img = render_state_to_image(state)
        else:
            print("Warning: Could not extract image from State object")
            # Create a blank image as fallback
            current_img = np.zeros((500, 500, 3), dtype=np.uint8)
    else:
        # Handle other types (pygame Surface, numpy array, etc.)
        current_img = convert_to_numpy_if_needed(state)
    
    # Ensure current_img is a numpy array
    if not isinstance(current_img, np.ndarray):
        print(f"Warning: current_img is not a numpy array, it's {type(current_img)}")
        current_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Load the target image
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(project_root, "data", f"{image_name}.jpg")
    target_img = cv2.imread(target_path)
    
    if target_img is None:
        print(f"Warning: Could not load target image from {target_path}")
        target_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Compute similarity
    similarity_score = compute_similarity_score(current_img, target_img)
    
    # Extract other features as needed
    # ...
    
    return similarity_score  # or return a feature vector

# Set up a dummy display before initializing pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
pygame.display.set_mode((1, 1))  # Create a small hidden display surface

def convert_to_numpy_if_needed(state):
    """Convert various types to numpy arrays for image processing"""
    if isinstance(state, np.ndarray):
        return state
    elif isinstance(state, pygame.Surface):
        # Convert pygame surface to numpy array
        array = pygame.surfarray.array3d(state)
        return array.transpose(1, 0, 2)  # Correct orientation
    elif hasattr(state, 'get_array'):
        # Some custom objects might have a method to get array
        return state.get_array()
    else:
        print(f"Warning: Unsupported state type: {type(state)}")
        return np.zeros((500, 500, 3), dtype=np.uint8)

def render_state_to_image(state):
    """Render a state with puzzle pieces to an image"""
    # Create a blank image
    width, height = 1200, 800  # Default size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(240)  # Light gray background
    
    # Draw the puzzle board
    box_x, box_y = 50, 50
    box_width, box_height = 800, 600  # Default size
    
    # Try to get actual dimensions if available
    if hasattr(state, 'box_width') and hasattr(state, 'box_height'):
        box_width, box_height = state.box_width, state.box_height
    
    # Draw board as gray rectangle
    cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (180, 180, 180), -1)
    
    # Draw pieces if available
    if hasattr(state, 'pieces'):
        for piece in state.pieces:
            if hasattr(piece, 'image') and hasattr(piece, 'current_pos'):
                # Convert pygame surface to numpy array if needed
                piece_img = convert_to_numpy_if_needed(piece.image)
                x, y = piece.current_pos
                
                # Paste the piece onto the image
                h, w = piece_img.shape[:2]
                image[y:y+h, x:x+w] = piece_img
    
    return image
