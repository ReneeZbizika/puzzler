#!/usr/bin/env python3
import os
import math
import argparse
import svgwrite
import cairosvg
import cv2
import numpy as np
from puzzle_generator import JigsawPuzzleGenerator
import json

###############################
# Convert SVG to PNG Function #
###############################

def convert_svg_to_png(input_svg, output_png):
    cairosvg.svg2png(url=input_svg, write_to=output_png)
    print(f"Converted {input_svg} to {output_png}")

###############################################
# Extract Puzzle Pieces Using Template Mask  #
###############################################

def extract_puzzle_pieces(template_png, original_png, output_folder, xn=15, yn=10, threshold=200, dilate_edges=True):
    """
    Uses the puzzle template PNG (white background, black edges) as a mask to extract each piece
    from the original image. The extracted pieces are saved in output_folder with transparent backgrounds.
    """
    template = cv2.imread(template_png, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template PNG: {template_png}")
    original = cv2.imread(original_png)
    if original is None:
        raise ValueError(f"Could not load original image: {original_png}")
    if template.shape[:2] != original.shape[:2]:
        # change og shape to match template
        original = cv2.resize(original, (template.shape[1], template.shape[0]))
        
    # Compute a dynamic minimum area threshold.
    h, w = template.shape[:2]
    total_area = w * h
    expected_piece_area = total_area / (xn * yn)
    
    min_area_factor = 0.3 #(only contours that cover at least 30% of the expected area will be kept.
    min_area = expected_piece_area * min_area_factor
    print(f"Template dimensions: {w}x{h}, total area: {total_area}, "
        f"expected piece area: {expected_piece_area:.2f}, "
        f"min_area threshold: {min_area:.2f}")

    ret, binary = cv2.threshold(template, threshold, 255, cv2.THRESH_BINARY)
    if dilate_edges:
        inv = 255 - binary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(inv, kernel, iterations=1)
        separated = 255 - dilated
    else:
        separated = binary
    contours, hierarchy = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} pieces.")
    os.makedirs(output_folder, exist_ok=True)
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            print(f"Ignoring contour {idx} with area {area} (below min_area threshold)")
            continue
    
        # Create a mask for this piece
        piece_mask = np.zeros_like(template)
        cv2.drawContours(piece_mask, [cnt], -1, 255, thickness=-1)
        
        # Create a 4-channel image (RGBA) with transparent background
        # First convert original to BGRA (add alpha channel)
        original_rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        
        # Create a transparent image of the same size
        piece_transparent = np.zeros_like(original_rgba)
        
        # Copy the RGB channels from the original image where the mask is non-zero
        piece_transparent[..., 0] = np.where(piece_mask == 255, original[..., 0], 0)
        piece_transparent[..., 1] = np.where(piece_mask == 255, original[..., 1], 0)
        piece_transparent[..., 2] = np.where(piece_mask == 255, original[..., 2], 0)
        
        # Set alpha channel to fully opaque (255) where the mask is non-zero, and fully transparent (0) elsewhere
        piece_transparent[..., 3] = np.where(piece_mask == 255, 255, 0)
        
        # Crop to bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        piece_cropped = piece_transparent[y:y+h, x:x+w]
        
        # Save with transparency
        out_path = os.path.join(output_folder, f"piece_{idx+1}.png")
        cv2.imwrite(out_path, piece_cropped)
        print(f"Saved piece {idx+1} to {out_path} with transparent background")

####################
# Main Entry Point #
####################

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Generate a jigsaw puzzle template, convert to PNG mask, and extract puzzle pieces from an original image."
    )
    parser.add_argument("original", help="Path to the original image (PNG).")
    parser.add_argument("--width", type=float, default=0, help="Width for the puzzle template. If 0, use the original image width.")
    parser.add_argument("--height", type=float, default=0, help="Height for the puzzle template. If 0, use the original image height.")
    parser.add_argument("--xn", type=int, default=5, help="Number of columns in the puzzle.")
    parser.add_argument("--yn", type=int, default=5, help="Number of rows in the puzzle.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for puzzle generation.")
    parser.add_argument("--tabsize", type=float, default=20, help="Tab size percentage.")
    parser.add_argument("--jitter", type=float, default=4, help="Jitter percentage.")
    parser.add_argument("--radius", type=float, default=2.0, help="Corner radius.")
    parser.add_argument("--output_template_svg", default="template.svg", help="Output filename for the SVG template.")
    parser.add_argument("--output_template_png", default="template.png", help="Output filename for the PNG template.")
    parser.add_argument("--output_pieces_folder", default="pieces", help="Folder to save extracted puzzle pieces.")
    parser.add_argument("--threshold", type=int, default=200, help="Threshold for binarizing the template.")
    parser.add_argument("--no-dilate", action="store_true", help="Disable dilation when extracting pieces.")
    parser.add_argument("--use_original_size", action="store_true",
                        help="Automatically use the original image dimensions for the puzzle template.")
    # Add new argument for scaling factor
    parser.add_argument("--scale_factor", type=float, default=0.25,
                        help="Scale factor for the final puzzle pieces (default: 0.25 = 25% of original size)")
    
    args = parser.parse_args()

    # If --use_original_size is set, read the original image to get its dimensions.
    if args.use_original_size:
        img = cv2.imread(args.original)
        if img is None:
            raise ValueError(f"Could not load original image: {args.original}")
        height, width = img.shape[:2]
        print(f"Using original image dimensions: width={width}, height={height}")
        args.width = width
        args.height = height
    else:
        if args.width == 0 or args.height == 0:
            raise ValueError("Please provide valid --width and --height values or use --use_original_size.")

    # Store original dimensions before scaling
    original_width = args.width
    original_height = args.height
    
    # Step 1: Generate the puzzle template as an SVG.
    generator = JigsawPuzzleGenerator(
        seed=args.seed,
        tab_size=args.tabsize,
        jitter=args.jitter,
        xn=args.xn,
        yn=args.yn,
        width=args.width,
        height=args.height,
        radius=args.radius
    )
    svg_data = generator.generate_svg()
    with open(args.output_template_svg, "w", encoding="utf-8") as f:
        f.write(svg_data)
    print(f"SVG puzzle template saved to {args.output_template_svg}")
    
    # Step 2: Convert the SVG template to a PNG.
    convert_svg_to_png(args.output_template_svg, args.output_template_png)
    
    # Step 3: Use the template PNG as a mask to extract pieces from the original image.
    extract_puzzle_pieces(args.output_template_png, args.original, args.output_pieces_folder,
                          xn=args.xn, yn=args.yn, threshold=args.threshold, dilate_edges=not args.no_dilate)
    
    # Step 4: Scale down the extracted pieces
    scale_puzzle_pieces(args.output_pieces_folder, args.scale_factor)
    
    # Construct the command string using parser arguments.
    # Flags like --no-dilate and --use_original_size are included if True.
    command = (
        f"python src/full_pipeline.py {args.original} "
        f"--width {args.width} --height {args.height} --xn {args.xn} --yn {args.yn} --seed {args.seed} "
        f"--tabsize {args.tabsize} --jitter {args.jitter} --radius {args.radius} "
        f"--output_template_svg {args.output_template_svg} --output_template_png {args.output_template_png} "
        f"--output_pieces_folder {args.output_pieces_folder} --threshold {args.threshold} "
        f"{'--no-dilate ' if args.no_dilate else ''}"
        f"{'--use_original_size ' if args.use_original_size else ''}"
        f"--scale_factor {args.scale_factor}"
    )
    
    # Step 5: Save the image parameters to a text file
    save_image_parameters(
        args.original,
        original_width,
        original_height,
        args.scale_factor,
        args.xn,
        args.yn, 
        comm=command
    )
    
    # Print the original and scaled dimensions
    scaled_width = int(original_width * args.scale_factor)
    scaled_height = int(original_height * args.scale_factor)
    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Scaled dimensions (after {args.scale_factor*100}% scaling): {scaled_width}x{scaled_height}")

# Add a new function to scale down the puzzle pieces
def scale_puzzle_pieces(pieces_folder, scale_factor):
    """
    Scale down all puzzle pieces in the given folder by the specified scale factor.
    Preserves transparency in the images.
    
    Args:
        pieces_folder: Folder containing the puzzle pieces
        scale_factor: Factor to scale the pieces by (e.g., 0.25 for 25% of original size)
    """
    if not os.path.exists(pieces_folder):
        print(f"Warning: Pieces folder {pieces_folder} does not exist.")
        return
    
    piece_files = [f for f in os.listdir(pieces_folder) if f.endswith('.png')]
    if not piece_files:
        print(f"Warning: No PNG files found in {pieces_folder}.")
        return
    
    print(f"Scaling {len(piece_files)} puzzle pieces by factor {scale_factor}...")
    
    for piece_file in piece_files:
        piece_path = os.path.join(pieces_folder, piece_file)
        
        # Load the piece image with alpha channel
        piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
        if piece_img is None:
            print(f"Warning: Could not load piece {piece_path}.")
            continue
        
        # Get original dimensions
        h, w = piece_img.shape[:2]
        
        # Calculate new dimensions
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Resize the image, preserving transparency
        resized_img = cv2.resize(piece_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save the resized image (overwrite the original)
        cv2.imwrite(piece_path, resized_img)
        
        print(f"Scaled {piece_file} from {w}x{h} to {new_w}x{new_h}")
    
    print(f"Finished scaling all puzzle pieces in {pieces_folder}")

    """
def save_image_parameters(original_image_path, width, height, scale_factor, xn, yn):
    """
    #Save image parameters to a text file in the params folder.
    #The file will be named <img_name>_params.txt
    """
    # Create params folder if it doesn't exist
    params_folder = "params"
    os.makedirs(params_folder, exist_ok=True)
    
    # Extract image name from the path
    img_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Create the parameter file path
    param_file_path = os.path.join(params_folder, f"{img_name}_params.txt")
    
    # Calculate scaled dimensions
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    # Write parameters to the file
    with open(param_file_path, 'w') as f:
        f.write(f"Image: {original_image_path}\n")
        f.write(f"Original dimensions: {width}x{height}\n")
        f.write(f"Scaled dimensions: {scaled_width}x{scaled_height}\n")
        f.write(f"Scale factor: {scale_factor}\n")
        f.write(f"Grid: {xn}x{yn}\n")
        f.write(f"Total pieces: {xn * yn}\n")
    
    print(f"Saved image parameters to {param_file_path}")
    
    """
    
def save_image_parameters(original_image_path, width, height, scale_factor, xn, yn, comm):
    """
    Save image parameters as JSON in the params folder.
    The JSON file will be named <img_name>_params.json.
    
    Saved parameters include:
      - Image: the path to the original image.
      - Original dimensions: the width and height of the original image.
      - Scaled dimensions: the dimensions of the image after scaling.
      - Scale factor: the factor used to scale the original dimensions.
      - Grid: the number of grid rows and columns (xn x yn).
      - Total pieces: computed as xn * yn.
      - BOX_WIDTH and BOX_HEIGHT: the dimensions of the puzzle's solution box.
      - SCREEN_WIDTH and SCREEN_HEIGHT: overall screen dimensions computed with margins.
      - Comm: the full command that was used, constructed from several parser arguments.
    """
    # Create params folder if it doesn't exist.
    params_folder = "params"
    os.makedirs(params_folder, exist_ok=True)
    
    # Extract the image name (without extension) from the path.
    img_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Build the JSON file path.
    param_file_path = os.path.join(params_folder, f"{img_name}_params.json")
    
    # Calculate the scaled dimensions.
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    # Define the solution box dimensions.
    # Here, we assume the solution box matches the scaled image dimensions.
    BOX_WIDTH = scaled_width * 3.8
    BOX_HEIGHT = scaled_height * 3.8
    
    # Define screen dimensions based on the box dimensions plus margins.
    margin = 100  # Adjust margin as needed.
    SCREEN_WIDTH = BOX_WIDTH * 1.5 + 2 * margin
    SCREEN_HEIGHT = BOX_HEIGHT * 1.5 + 2 * margin
    
    # Prepare the data dictionary.
    data = {
        "Image": original_image_path,
        "Original dimensions": f"{width}x{height}",
        "Scaled dimensions": f"{scaled_width}x{scaled_height}",
        "Scale factor": scale_factor,
        "Grid": f"{xn}x{yn}",
        "Total pieces": xn * yn,
        "BOX_WIDTH": BOX_WIDTH,
        "BOX_HEIGHT": BOX_HEIGHT,
        "SCREEN_WIDTH": SCREEN_WIDTH,
        "SCREEN_HEIGHT": SCREEN_HEIGHT, 
        "Command": comm
    }
    
    # Save the data dictionary as JSON.
    with open(param_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved image parameters to {param_file_path}")

# Example run example FROM SRC FOLDER
# python src/full_pipeline.py data/img_2.jpg --xn 5 --yn 5 --seed 1234 --output_pieces_folder pieces_img_2 --use_original_size --scale_factor 0.05

if __name__ == "__main__":
    main()