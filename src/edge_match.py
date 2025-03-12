# Edge Compatability Algorithm 
# Inspired by https://www-users.cse.umn.edu/~olver/v_/puzzles.pdf

import numpy as np
import cv2
import math
import scipy as sp
import itertools
import os

# Helper functions
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


# Compute a discrete “signature” that approximate the curvature and its derivative along the edge
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

# Define an edge compatibility score as a function of how “close” these signature vectors are
# Compatability score between 2 edges
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

if __name__ == "__main__":
    folder_path = "pieces_img_2"
    
    # Extract edges and compute compatibility
    edge_signatures, edge_to_piece_map, image_files = extract_edges_from_pieces(folder_path)
    compatibility_matrix = compute_compatibility_matrix(edge_signatures)

    # Save and display results
    np.savetxt("edge_compatibility_matrix.csv", compatibility_matrix, delimiter=",", fmt="%.4f")

    print("Edge-to-Piece Mapping:", edge_to_piece_map)
    print("Edge Compatibility Matrix:\n", compatibility_matrix)

# DEBUGGING USE 
"""
if __name__ == "__main__":
    image_path = "pieces_img_2/piece_4.png"
    contour_points, thresh_img = segment_piece_black_bg(image_path)
    
    print("Number of points in contour:", len(contour_points))
    # Optionally save the thresholded image for debugging
    cv2.imwrite("thresh_debug.png", thresh_img)
"""