"""
Modified version of pyjigsaw to create puzzle pieces with transparent backgrounds.

This module extends functionality from the original pyjigsaw library
but modifies the image embedding process to support transparency.
"""

import base64
import os
import subprocess
import tempfile
from math import ceil
import random
from PIL import Image
from svgpathtools import svg2paths
import numpy as np
import cv2
import io
from pathlib import Path
import glob
from pyjigsaw import jigsawfactory
import re

class Cut:
    """
    Used to create the cutting of the jigsaw.
    Defines the number of piece in width and height.
    """

    def __init__(self, pieces_height, pieces_width, abs_height=None, abs_width=None, 
                 image=None, use_image=False, stroke_color="black", fill_color="white"):
        """
        Constructor.
        :param pieces_height: number of pieces in height.
        :param pieces_width: number of pieces in width.
        :param abs_height: absolute height of final jigsaw. If None, value is pieces_height * 100.
        :param abs_width: absolute width of final jigsaw. If None, value is pieces_width * 100.
        :param image: image file to use instead of solid color bg, defaults to None
        :param use_image: weather to use passed image, defaults to False
        :param stroke_color: color for the resulting svg stroke, defaults to "black"
        :param fill_color: color for the resulting svg fill, defaults to "white"
        """

        # Number of pieces in height and width.
        self.pieces_height = int(pieces_height)
        self.pieces_width = int(pieces_width)
        # Absolute width and height of final jigsaw.
        self.abs_height = abs_height
        self.abs_width = abs_width
        # Final width and height of each piece.
        self.piece_height = None  # Computed later
        self.piece_width = None  # Computed later
        # Cutting template
        self.cut_template = None
        self.image = image
        self.use_image = use_image
        self.stroke_color = stroke_color
        self.fill_color = fill_color  # This is now transparent
        # Update and compute all internal attributes.
        self.update_cut_template()

    def update_cut_template(self):
        """
        Updates the internal attributes of this instance.
        Called upon initialization.
        """
        # Some default values.
        if self.abs_height is None:
            self.abs_height = self.pieces_height * 100
        if self.abs_width is None:
            self.abs_width = self.pieces_width * 100

        # Size of each piece.
        self.piece_height = self.abs_height / self.pieces_height
        self.piece_width = self.abs_width / self.pieces_width

        # Cutting template.
        self.cut_template = [[None for _ in range(self.pieces_width)]
                                for _ in range(self.pieces_height)]


def extract_average_colors(image_path, num_rows, num_cols):
    """
    Extract the average color from different sections of an image
    
    Args:
        image_path (str): Path to the input image
        num_rows (int): Number of rows to divide the image into
        num_cols (int): Number of columns to divide the image into
        
    Returns:
        list: List of RGB color tuples for each section
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Calculate section dimensions
    section_height = height // num_rows
    section_width = width // num_cols
    
    # Extract average colors
    colors = []
    for row in range(num_rows):
        for col in range(num_cols):
            # Define section boundaries
            top = row * section_height
            bottom = (row + 1) * section_height if row < num_rows - 1 else height
            left = col * section_width
            right = (col + 1) * section_width if col < num_cols - 1 else width
            
            # Extract section
            section = img[top:bottom, left:right]
            
            # Calculate average color
            avg_color = np.mean(section, axis=(0, 1)).astype(int)
            
            # Convert to RGB hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
            colors.append(hex_color)
    
    return colors


def make_image_transparent(image_path, output_path=None):
    """
    Convert white background to transparent for an image
    
    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save output image. If None, creates a temporary file.
        
    Returns:
        str: Path to the output image with transparent background
    """
    try:
        # Use PIL for better color handling
        from PIL import Image
        
        # Open the image with PIL
        img = Image.open(image_path).convert("RGBA")
        
        # Get the data
        datas = img.getdata()
        
        # Create a new image with transparent background
        new_data = []
        for item in datas:
            # If the pixel is whitish (threshold can be adjusted)
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                # Make it transparent (keeping RGB values but setting alpha to 0)
                new_data.append((item[0], item[1], item[2], 0))
            else:
                # Keep the original color with full opacity
                new_data.append(item)
                
        # Update the image with the new data
        img.putdata(new_data)
        
        # If output path is not specified, use temporary file
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.png')
        
        # Save the image with transparency
        img.save(output_path, "PNG")
        
        return output_path
    
    except Exception as e:
        print(f"Error making image transparent: {e}")
        # Return the original image if there's an error
        return image_path


def image_encode(original_image, make_transparent=True):
    """
    Encode an image as base64 for embedding in SVG, with option to make it transparent
    
    Args:
        original_image (str): Path to the image
        make_transparent (bool): Whether to make white background transparent
        
    Returns:
        tuple: (file_extension, base64_encoded_data)
    """
    try:
        if make_transparent:
            # Create a temporary transparent version
            temp_transparent = make_image_transparent(original_image)
            img_path = temp_transparent
            file_ext = "png"  # Always use PNG for transparent images
        else:
            img_path = original_image
            # Get the file extension from the original image
            file_ext = os.path.splitext(original_image)[1][1:].lower()
            # Convert jpg to jpeg for MIME type compatibility
            file_ext = "jpeg" if file_ext == "jpg" else file_ext
        
        # Read the image binary data
        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
            
        # Convert to base64
        encoded_data = base64.b64encode(img_data).decode('utf-8')
        
        # Clean up temporary file if created
        if make_transparent and temp_transparent != original_image:
            os.unlink(temp_transparent)
            
        return (file_ext, encoded_data)
    except Exception as e:
        print(f"Error encoding image: {e}")
        # Fallback to original image if there's an error
        with open(original_image, 'rb') as img_file:
            encoded_data = base64.b64encode(img_file.read()).decode('utf-8')
            file_ext = os.path.splitext(original_image)[1][1:].lower()
            file_ext = "jpeg" if file_ext == "jpg" else file_ext
            return (file_ext, encoded_data)


class JigsawWrapper:
    """
    Wrapper for the Jigsaw class from pyjigsaw.jigsawfactory
    This class enhances the original with the ability to create transparent SVG puzzle pieces
    """
    
    def __init__(self, jigsaw_obj):
        """
        Initialize with an existing Jigsaw object
        
        Args:
            jigsaw_obj: A pyjigsaw.jigsawfactory.Jigsaw instance
        """
        self.jigsaw = jigsaw_obj
        self.original_image_encode = None
        self._patch_image_encode()
    
    def _patch_image_encode(self):
        """
        Replace the image_encode function in pyjigsaw with our version
        that supports transparency
        """
        # Store the original function for later restoration
        import pyjigsaw.jigsawfactory
        self.original_image_encode = pyjigsaw.jigsawfactory.image_encode
        # Replace with our function
        pyjigsaw.jigsawfactory.image_encode = image_encode
    
    def _restore_image_encode(self):
        """
        Restore the original image_encode function
        """
        if self.original_image_encode:
            import pyjigsaw.jigsawfactory
            pyjigsaw.jigsawfactory.image_encode = self.original_image_encode
    
    def generate_svg_jigsaw(self, outdirectory):
        """
        Generate SVG jigsaw pieces with transparent backgrounds
        
        Args:
            outdirectory (str): Directory to save the SVG files
        """
        try:
            # Make sure the output directory exists
            if not os.path.exists(outdirectory):
                os.makedirs(outdirectory)
            
            # Store original values to restore later
            original_img_path = None
            original_fill = None
            
            if self.jigsaw.cut.use_image and hasattr(self.jigsaw.cut, 'image') and self.jigsaw.cut.image:
                # Remember the original image path
                original_img_path = self.jigsaw.cut.image
                
                # We'll let the original function process the image
                # Our patched image_encode will handle transparency
            
            # Remember the original fill color
            if hasattr(self.jigsaw.cut, 'fill_color'):
                original_fill = self.jigsaw.cut.fill_color
                # We now set fill to transparent, not "none"
                # This ensures we keep the original image visible
                self.jigsaw.cut.fill_color = "transparent"
            
            # Call the original generate_svg_jigsaw method with our modifications active
            result = self.jigsaw.generate_svg_jigsaw(outdirectory)
            
            # Restore original settings
            if original_img_path:
                self.jigsaw.cut.image = original_img_path
            
            if original_fill:
                self.jigsaw.cut.fill_color = original_fill
            
            # After generation, modify the SVG files to ensure they have transparent backgrounds
            self._post_process_svg_files(outdirectory)
            
            return result
        
        finally:
            # Restore the original image_encode function
            self._restore_image_encode()
    
    def _post_process_svg_files(self, directory):
        """
        Post-process SVG files to ensure transparency is preserved
        while keeping the image content visible
        """
        # Process each SVG file
        for filename in os.listdir(directory):
            if filename.endswith('.svg'):
                file_path = os.path.join(directory, filename)
                
                # Read the SVG file
                with open(file_path, 'r') as f:
                    svg_content = f.read()
                
                # Ensure the SVG has a transparent background
                if 'style="background-color: transparent;"' not in svg_content:
                    svg_content = svg_content.replace('<svg', '<svg style="background-color: transparent;"', 1)
                
                # We DO NOT want to set fill="none" for paths that contain the image
                # Instead, we'll ensure the clipping path works properly
                
                # Check if there's a clipPath
                if '<clipPath id="crop">' in svg_content:
                    # Make sure it's properly defined with no extra attributes that might interfere
                    clip_path_pattern = r'(<clipPath id="crop">)(.*?)(<\/clipPath>)'
                    
                    def clean_clip_path(match):
                        # Get the content inside the clipPath
                        before, content, after = match.groups()
                        # Make sure the path has no fill or other attributes that might interfere
                        if '<path' in content:
                            # Keep the path data but remove any fill or other styling that might interfere
                            path_content = re.sub(r'<path([^>]*?)fill="[^"]*"', r'<path\1', content)
                            path_content = path_content.replace('stroke="none"', '')
                            return before + path_content + after
                        return match.group(0)
                    
                    svg_content = re.sub(clip_path_pattern, clean_clip_path, svg_content, flags=re.DOTALL)
                
                # Make sure the image element properly references the clip-path
                if '<image' in svg_content and 'clip-path=' not in svg_content:
                    svg_content = re.sub(r'<image', r'<image clip-path="url(#crop)"', svg_content)
                    
                # White background rectangles should be transparent
                svg_content = re.sub(r'<rect([^>]*)fill="white"', r'<rect\1fill="transparent"', svg_content)
                
                # Write back the modified content
                with open(file_path, 'w') as f:
                    f.write(svg_content)

def wrap_jigsaw(jigsaw_obj):
    """
    Create a wrapper around a pyjigsaw.jigsawfactory.Jigsaw object
    that enhances it with transparent background capabilities
    
    Args:
        jigsaw_obj: A pyjigsaw.jigsawfactory.Jigsaw instance
        
    Returns:
        JigsawWrapper: Enhanced jigsaw generator
    """
    return JigsawWrapper(jigsaw_obj) 