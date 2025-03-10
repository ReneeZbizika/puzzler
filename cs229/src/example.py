#!/usr/bin/env python3
"""
Example script for using the puzzle generator.
This shows how to call the generate_jigsaw_svgs function directly from your own code.
"""

import os
import sys
from puzzlegen import generate_jigsaw_svgs

def main():
    # Get the absolute path to the cs229 directory (parent of src)
    cs229_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Example usage of the puzzle generator
    # Try to use a sample image from the data directory
    data_dir = os.path.join(cs229_dir, 'data')
    output_dir = os.path.join(cs229_dir, 'output', 'example_puzzle')
    
    # Check if there are any images in the data directory
    image_files = [f for f in os.listdir(data_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        # Use the first image found
        image_path = os.path.join(data_dir, image_files[0])
        print(f"Using sample image: {image_path}")
    else:
        print("No sample images found in the data directory.")
        print("Please add an image to the 'data' directory and try again.")
        return
    
    # Generate a 3x4 puzzle with blue outlines
    piece_svg_files = generate_jigsaw_svgs(
        image_path=image_path,
        output_dir=output_dir,
        rows=3,
        cols=4,
        stroke_color="blue",
        fill_color="black"
    )
    
    # Print all generated files
    print("\nGenerated files:")
    for svg_file in piece_svg_files:
        print(f"- {svg_file}")
    
    print(f"\nTotal pieces: {len(piece_svg_files)}")
    print(f"Puzzle pieces saved to: {output_dir}")

if __name__ == "__main__":
    main() 