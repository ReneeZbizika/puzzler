#!/usr/bin/env python3
"""
Clean SVG Puzzle Pieces

This script processes SVG files to remove backgrounds and whitespace,
resulting in clean, transparent puzzle pieces.
"""

import os
import re
import glob
import argparse
import xml.etree.ElementTree as ET

def clean_svg(svg_path, output_path=None, verbose=False):
    """
    Clean an SVG file by:
    1. Removing background rectangles
    2. Setting fill to transparent
    3. Optimizing viewBox
    
    Args:
        svg_path: Path to the input SVG file
        output_path: Path to save the cleaned SVG file (if None, overwrites the original)
        verbose: Whether to print detailed information
    
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"Processing {os.path.basename(svg_path)}...")
    
    try:
        # Register the SVG namespace
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        
        # Parse the SVG file
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Track if we made changes
        changes_made = False
        
        # 1. Remove background rectangles (usually the first one)
        for rect in root.findall('.//{http://www.w3.org/2000/svg}rect'):
            # Background rectangles usually have x=0, y=0 and large width/height
            if 'x' in rect.attrib and 'y' in rect.attrib and rect.attrib.get('x', '') == '0' and rect.attrib.get('y', '') == '0':
                parent = list(root.iter())[0]  # Get the top-level element
                parent.remove(rect)
                changes_made = True
                if verbose:
                    print("  - Removed background rectangle")
                break  # Only remove the first matching rectangle
        
        # 2. Set fill to transparent for all elements
        fill_count = 0
        for element in root.findall('.//*[@fill]'):
            if element.attrib.get('fill') != 'transparent':
                element.attrib['fill'] = 'transparent'
                fill_count += 1
                changes_made = True
        
        if verbose and fill_count > 0:
            print(f"  - Set {fill_count} elements to transparent fill")
        
        # 3. Try to adjust viewBox to focus on the path elements (puzzle piece)
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        if paths and hasattr(paths[0], 'attrib') and 'd' in paths[0].attrib:
            # Find bounding coordinates from path data
            # This is a simplified approach - for precise bounds you'd need a path parser
            path_data = paths[0].attrib['d']
            
            # Extract all coordinates from the path (simplified)
            coords = re.findall(r'[ML]\s*([\d.-]+)[, ]([\d.-]+)', path_data)
            
            if coords:
                # Convert to floats and find min/max values
                x_coords = [float(x) for x, y in coords]
                y_coords = [float(y) for x, y in coords]
                
                min_x = min(x_coords)
                min_y = min(y_coords)
                max_x = max(x_coords)
                max_y = max(y_coords)
                
                # Add some padding (5%)
                width = max_x - min_x
                height = max_y - min_y
                padding_x = width * 0.05
                padding_y = height * 0.05
                
                min_x -= padding_x
                min_y -= padding_y
                max_x += padding_x
                max_y += padding_y
                
                # Update viewBox
                viewbox = f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}"
                root.attrib['viewBox'] = viewbox
                
                # Update width and height attributes if present
                if 'width' in root.attrib and 'height' in root.attrib:
                    root.attrib['width'] = str(max_x - min_x)
                    root.attrib['height'] = str(max_y - min_y)
                
                if verbose:
                    print(f"  - Updated viewBox to: {viewbox}")
                changes_made = True
        
        # Save the file
        output_path = output_path or svg_path
        tree.write(output_path)
        
        if not changes_made:
            if verbose:
                print("  - No changes needed")
            return True
        
        if verbose:
            print(f"  - Saved to {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        if verbose:
            print(f"  - Error: {str(e)}")
        return False

def process_directory(input_dir, output_dir=None, verbose=False):
    """
    Process all SVG files in a directory
    
    Args:
        input_dir: Directory containing SVG files
        output_dir: Directory to save cleaned SVG files (if None, overwrites originals)
        verbose: Whether to print detailed information
    
    Returns:
        Number of successfully processed files
    """
    # Find all SVG files
    svg_files = glob.glob(os.path.join(input_dir, "*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {input_dir}")
        return 0
    
    print(f"Found {len(svg_files)} SVG files in {input_dir}")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each file
    success_count = 0
    for svg_file in svg_files:
        if output_dir:
            output_path = os.path.join(output_dir, os.path.basename(svg_file))
        else:
            output_path = svg_file
            
        if clean_svg(svg_file, output_path, verbose):
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(svg_files)} SVG files")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Clean SVG puzzle pieces by removing backgrounds and whitespace")
    parser.add_argument("input_path", help="Path to input SVG file or directory")
    parser.add_argument("--output", "-o", help="Path to output SVG file or directory (if omitted, overwrites originals)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        # Process directory
        process_directory(args.input_path, args.output, args.verbose)
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith('.svg'):
        # Process single file
        if clean_svg(args.input_path, args.output, args.verbose):
            print("SVG file cleaned successfully")
        else:
            print("Failed to clean SVG file")
    else:
        print("Input path must be a directory or an SVG file")

if __name__ == "__main__":
    main() 