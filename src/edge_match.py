# Edge Compatability Algorithm 
# inspired by https://www-users.cse.umn.edu/~olver/v_/puzzles.pdf

import numpy as np
import cv2
import math

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
    
    print("Min pixel value:", np.min(color_img))
    print("Max pixel value:", np.max(color_img))
    print("Mean pixel value:", np.mean(color_img))
    print(color_img.shape)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print("Loaded image shape:", image.shape)
    #print(color_img[100, 100], color_img[200, 200])

    
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
    interpolator = interp1d(t_original, signature, axis=0, kind='cubic', fill_value="extrapolate")
    resampled_signature = interpolator(t_new)
    return resampled_signature

# Replace with actual interpolation code using numpy/scipy


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


def compute_compatibility_matrix(edge_signatures, alpha=1.0): 
    """
    Builds a compatibility matrix that holds the score for every pair of edges.

    Params:
    edge_signatures (list): list of edge signatures. 
    alpha (int or float): constant for threshhold

    Returns:
    matrix: Compatibility matrix where every entry (i, j) is the compatibility score between edge i and edge j.
    """
    num_edges = len(edge_signatures) 
    comp_matrix = np.zeros((num_edges, num_edges)) 
    for i in range(num_edges): 
        for j in range(num_edges): # Set diagonal entries to 0 (or 1) depending on your convention. 
            if i == j: 
                comp_matrix[i, j] = 0 # or leave it as is 
            else: 
                comp_matrix[i, j] = edge_compatibility(edge_signatures[i], edge_signatures[j], alpha) 
        
    return comp_matrix


# Pipeline
def pipeline(image_A_path, image_B_path, num_resample_points=100, alpha=0.1): 
    """ Pipeline to load two puzzle piece images, segment their contours, compute and 
    resample their edge signatures, and output an edge compatibility score. """ 
    # Load images. 
    image_A = load_image(image_A_path) 
    image_B = load_image(image_B_path)
    # Segment the puzzle pieces to extract their contours.
    contour_A = segment_piece(image_A)
    contour_B = segment_piece(image_B)

    # Compute the discrete edge signature for each contour.
    signature_A = compute_signature(contour_A)
    signature_B = compute_signature(contour_B)

    # Resample the signatures to have a fixed number of points.
    resampled_A = resample_edge(signature_A, num_resample_points)
    resampled_B = resample_edge(signature_B, num_resample_points)

    # Optionally, if the starting points of the signatures are arbitrary,
    # one could perform cyclic shifting here to find the best alignment.
    # For now, we simply compare them as is.

    # Compute the compatibility score.
    score = edge_compatibility(resampled_A, resampled_B, alpha=alpha)

    return score

# Example usage
#def main ():

if __name__ == "__main__":
    image_path = "pieces/piece_2.png"
    contour_points, thresh_img = segment_piece_black_bg(image_path)
    
    print("Number of points in contour:", len(contour_points))
    # Optionally save the thresholded image for debugging
    cv2.imwrite("thresh_debug.png", thresh_img)
    
"""if __name__ == "__main__":
    #main()
    image_A_path = "pieces/piece_2.png"  # Path to puzzle piece A image. 
    image_B_path = "pieces/piece_3.png" # Path to puzzle piece B image.

    score = pipeline(image_A_path, image_B_path, num_resample_points=100, alpha=0.1)
    print("Edge Compatibility Score:", score)"""