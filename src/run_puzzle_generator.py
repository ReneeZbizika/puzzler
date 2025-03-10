#!/usr/bin/env python3
"""
Main script to run the jigsaw puzzle generator.
This script provides a simple interface to the puzzle generator.
"""

import os
import sys
import argparse
import re
import xml.etree.ElementTree as ET
import glob

# Import the generate_jigsaw_svgs function from our puzzlegen module
from puzzlegen import generate_jigsaw_svgs

def clean_svg(svg_path, verbose=False):
    """
    Clean an SVG file by removing the white background while
    preserving the puzzle piece appearance.
    
    Args:
        svg_path: Path to the input SVG file
        verbose: Whether to print detailed information
    
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"Cleaning {os.path.basename(svg_path)}...")
    
    try:
        # Read the SVG file as text
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        changes_made = False
        
        # Find and remove any white background rectangles
        # Pattern to match rectangle elements with white fill
        rect_pattern = r'<rect[^>]*?fill="white"[^>]*?>'
        if re.search(rect_pattern, svg_content):
            # Remove rectangle elements with white fill
            svg_content = re.sub(rect_pattern, '', svg_content)
            changes_made = True
            if verbose:
                print("  - Removed white background rectangles")
        
        # Set the SVG background to be transparent
        if '<svg' in svg_content:
            # Check if SVG already has style attribute
            if 'style=' in svg_content and 'background-color: transparent' not in svg_content:
                # Add background-color: transparent to existing style
                svg_content = re.sub(r'<svg([^>]*?)style="([^"]*?)"', 
                                     r'<svg\1style="\2; background-color: transparent;"', 
                                     svg_content)
                changes_made = True
                if verbose:
                    print("  - Added transparent background to existing style")
            elif 'style=' not in svg_content:
                # Add style attribute with background-color: transparent
                svg_content = re.sub(r'<svg', 
                                     r'<svg style="background-color: transparent;"', 
                                     svg_content)
                changes_made = True
                if verbose:
                    print("  - Added transparent background style")
        
        # Write the updated content back to the file
        if changes_made:
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            if verbose:
                print("  ✓ SVG updated successfully")
            return True
        else:
            if verbose:
                print("  ✓ No changes needed")
            return False
            
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
        return False


def clean_svg_files(svg_files, verbose=False):
    """
    Clean multiple SVG files.
    
    Args:
        svg_files: List of paths to SVG files
        verbose: Whether to print detailed information
    """
    success_count = 0
    for svg_file in svg_files:
        if clean_svg(svg_file, verbose):
            success_count += 1
    
    print(f"Cleaned {success_count} out of {len(svg_files)} SVG files")


def main():
    # Get the absolute path to the cs229 directory (parent of src)
    cs229_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Define the output directory within cs229
    default_output_dir = os.path.join(cs229_dir, 'output')
    
    parser = argparse.ArgumentParser(description="Generate jigsaw puzzle pieces from an image")
    parser.add_argument("image_path", nargs="?", help="Path to the input image file (if not provided, will look in data directory)")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to output puzzle pieces")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows in the puzzle")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns in the puzzle")
    parser.add_argument("--stroke-color", default="red", help="Stroke color for the puzzle pieces")
    parser.add_argument("--fill-color", default="black", help="Fill color for the puzzle pieces")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning SVGs to remove backgrounds and whitespace")
    parser.add_argument("--clean-only", action="store_true", help="Only clean existing SVGs without generating new ones")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information during cleaning")
    parser.add_argument("--example", action="store_true", help="Run the example script instead")
    parser.add_argument("--all", action="store_true", help="Process all images in the data directory")
    parser.add_argument("--list", action="store_true", help="List all available images in the data directory")
    parser.add_argument("--vector-mode", action="store_true", help="Use vector-based puzzle generation (default)")
    args = parser.parse_args()

    if args.example:
        print("Running the example script...")
        from example import main as example_main
        example_main()
        return

    # If clean-only mode, just clean existing SVGs in the output directory
    if args.clean_only:
        svg_files = glob.glob(os.path.join(args.output_dir, "*.svg"))
        if not svg_files:
            print(f"No SVG files found in {args.output_dir}")
            return
        clean_svg_files(svg_files, args.verbose)
        return

    # Path to the data directory
    data_dir = os.path.join(cs229_dir, 'data')
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist. Creating it now.")
        os.makedirs(data_dir)
        print(f"Please add images to the '{data_dir}' directory and try again.")
        sys.exit(1)
    
    # Get all image files in the data directory
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                  
    # Sort the image files for consistent behavior
    image_files.sort()
    
    if args.list:
        if image_files:
            print("Available images in the data directory:")
            for i, img in enumerate(image_files, 1):
                print(f"{i}. {img}")
        else:
            print("No images found in the data directory.")
        return
        
    # If the user wants to process all images
    if args.all:
        if not image_files:
            print("No images found in the data directory.")
            return
            
        # Process each image file
        for img_file in image_files:
            img_path = os.path.join(data_dir, img_file)
            output_subdir = os.path.join(args.output_dir, os.path.splitext(img_file)[0])
            os.makedirs(output_subdir, exist_ok=True)
            
            print(f"\nProcessing {img_file}...")
            
            # Generate puzzle pieces
            piece_svgs = generate_jigsaw_svgs(
                image_path=img_path,
                output_dir=output_subdir,
                rows=args.rows,
                cols=args.cols,
                stroke_color=args.stroke_color,
                fill_color=args.fill_color
            )
            
            # Clean the SVGs if requested
            if not args.no_clean:
                clean_svg_files(piece_svgs, args.verbose)
                
        print("\nAll images processed successfully!")
        return
    
    # If a specific image is provided, process it
    if args.image_path:
        # Check if the image exists
        if not os.path.exists(args.image_path):
            # Try looking in the data directory
            data_img_path = os.path.join(data_dir, os.path.basename(args.image_path))
            if os.path.exists(data_img_path):
                args.image_path = data_img_path
            else:
                print(f"Error: Image file '{args.image_path}' not found.")
                sys.exit(1)
    else:
        # If no image is provided, use the first image in the data directory
        if not image_files:
            print("No images found in the data directory.")
            print(f"Please add images to the '{data_dir}' directory or specify an image path.")
            sys.exit(1)
        args.image_path = os.path.join(data_dir, image_files[0])
        print(f"No image specified, using '{os.path.basename(args.image_path)}'")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the puzzle pieces
    print(f"Generating puzzle pieces from {args.image_path}...")
    print(f"Using {args.rows} rows and {args.cols} columns")
    print(f"Output directory: {args.output_dir}")
    
    piece_svgs = generate_jigsaw_svgs(
        image_path=args.image_path,
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        stroke_color=args.stroke_color,
        fill_color=args.fill_color
    )
    
    # Clean the SVGs if requested
    if not args.no_clean:
        clean_svg_files(piece_svgs, args.verbose)
    
    print(f"\nPuzzle generation complete! {len(piece_svgs)} puzzle pieces generated.")
    

if __name__ == "__main__":
    main() 