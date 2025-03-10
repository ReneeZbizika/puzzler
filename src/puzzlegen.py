# -*- coding: utf-8 -*-
"""puzzlegen.py

A script to generate jigsaw puzzle pieces from an image.
Originally from a Google Colab notebook.
"""

# Required imports
import os
import re
import glob
import argparse
import sys
import subprocess
# Import the original pyjigsaw and our wrapper
from pyjigsaw import jigsawfactory
from vectorjigsaw import wrap_jigsaw
from PIL import Image

def generate_jigsaw_svgs(
    image_path: str,
    output_dir: str,
    rows: int,
    cols: int,
    stroke_color: str = "red",
    fill_color: str = "black"  # Default fill color for pyjigsaw
):
    """
    Generates jigsaw puzzle pieces from the input image as individual SVG files
    in 'output_dir'. Returns a list of those piece file paths.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str
        Directory to save the generated SVG files
    rows : int
        Number of rows in the puzzle
    cols : int
        Number of columns in the puzzle
    stroke_color : str
        Color of the puzzle piece outlines
    fill_color : str
        Fill color for the puzzle pieces
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create the puzzle cut using original pyjigsaw
    cut = jigsawfactory.Cut(
        rows,
        cols,
        image=image_path,
        use_image=True,
        stroke_color=stroke_color,
        fill_color=fill_color
    )
    
    # Create the jigsaw using original pyjigsaw
    pyjigsaw_puzzle = jigsawfactory.Jigsaw(cut, image_path)
    
    # Wrap the pyjigsaw puzzle with our transparent background handler
    puzzle = wrap_jigsaw(pyjigsaw_puzzle)

    # Generate all piece SVGs in output_dir with transparent backgrounds
    puzzle.generate_svg_jigsaw(output_dir)

    # Collect the generated files
    piece_svg_files = sorted(glob.glob(os.path.join(output_dir, "*.svg")))
    print(f"Generated {len(piece_svg_files)} piece SVGs in '{output_dir}'")
    
    # Print information about the puzzle pieces
    print("\nNote: The puzzle pieces now have transparent backgrounds.")
    print("Each SVG uses a clip-path with the randomized puzzle piece shape")
    print("and the original image with white pixels made transparent.")
    
    return piece_svg_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate jigsaw puzzle pieces from an image")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("--output-dir", default="../output", help="Directory to output puzzle pieces")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows in the puzzle")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns in the puzzle")
    parser.add_argument("--stroke-color", default="red", help="Stroke color for the puzzle pieces")
    parser.add_argument("--fill-color", default="black", help="Fill color for the puzzle pieces")
    args = parser.parse_args()

    # Generate the puzzle pieces
    print(f"Generating puzzle pieces from {args.image_path}...")
    print(f"Using {args.rows} rows and {args.cols} columns")
    piece_svgs = generate_jigsaw_svgs(
        image_path=args.image_path,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        stroke_color=args.stroke_color,
        fill_color=args.fill_color
    )
    print(f"Successfully generated {len(piece_svgs)} puzzle pieces!")
    print(f"Puzzle pieces saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 