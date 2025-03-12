#!/usr/bin/env python3
import os
import math
import argparse
import svgwrite
import cairosvg
import cv2
import numpy as np

########################################
# JigsawPuzzleGenerator Implementation #
########################################

class JigsawPuzzleGenerator:
    def __init__(self, seed=1, tab_size=20, jitter=4, xn=15, yn=10,
                 width=300, height=200, radius=2.0):
        # Parameters from the HTML version.
        self.seed = float(seed)
        self.t = float(tab_size) / 200.0
        self.j = float(jitter) / 100.0
        self.xn = int(xn)
        self.yn = int(yn)
        self.width = float(width)
        self.height = float(height)
        self.radius = float(radius)
        # For final output, use offset = 0
        self.offset = 0.0

        # Internal state for generating piece curves.
        self.a = self.b = self.c = self.d = self.e = 0.0
        self.flip = True
        self.xi = self.yi = 0
        self.vertical = False

    # --- Random number functions (mimic the JS code) ---
    def js_random(self):
        x = math.sin(self.seed) * 10000
        self.seed += 1
        return x - math.floor(x)

    def uniform(self, min_val, max_val):
        r = self.js_random()
        return min_val + r * (max_val - min_val)

    def rbool(self):
        return self.js_random() > 0.5

    # --- Curve control point setup ---
    def first(self):
        self.e = self.uniform(-self.j, self.j)
        self.next()

    def next(self):
        flipold = self.flip
        self.flip = self.rbool()
        if self.flip == flipold:
            self.a = -self.e
        else:
            self.a = self.e
        self.b = self.uniform(-self.j, self.j)
        self.c = self.uniform(-self.j, self.j)
        self.d = self.uniform(-self.j, self.j)
        self.e = self.uniform(-self.j, self.j)

    # --- Helpers for computing segment positions ---
    def sl(self):
        return self.height / self.yn if self.vertical else self.width / self.xn

    def sw(self):
        return self.width / self.xn if self.vertical else self.height / self.yn

    def ol(self):
        return self.offset + self.sl() * (self.yi if self.vertical else self.xi)

    def ow(self):
        return self.offset + self.sw() * (self.xi if self.vertical else self.yi)

    def l(self, v):
        ret = self.ol() + self.sl() * v
        return round(ret, 2)

    def w(self, v):
        multiplier = -1.0 if self.flip else 1.0
        ret = self.ow() + self.sw() * v * multiplier
        return round(ret, 2)

    # --- Functions for control points ---
    def p0l(self): return self.l(0.0)
    def p0w(self): return self.w(0.0)
    def p1l(self): return self.l(0.2)
    def p1w(self): return self.w(self.a)
    def p2l(self): return self.l(0.5 + self.b + self.d)
    def p2w(self): return self.w(-self.t + self.c)
    def p3l(self): return self.l(0.5 - self.t + self.b)
    def p3w(self): return self.w(self.t + self.c)
    def p4l(self): return self.l(0.5 - 2.0 * self.t + self.b - self.d)
    def p4w(self): return self.w(3.0 * self.t + self.c)
    def p5l(self): return self.l(0.5 + 2.0 * self.t + self.b - self.d)
    def p5w(self): return self.w(3.0 * self.t + self.c)
    def p6l(self): return self.l(0.5 + self.t + self.b)
    def p6w(self): return self.w(self.t + self.c)
    def p7l(self): return self.l(0.5 + self.b + self.d)
    def p7w(self): return self.w(-self.t + self.c)
    def p8l(self): return self.l(0.8)
    def p8w(self): return self.w(self.e)
    def p9l(self): return self.l(1.0)
    def p9w(self): return self.w(0.0)

    # --- Generate horizontal puzzle paths ---
    def gen_dh(self):
        self.vertical = False
        path = ""
        for yi in range(1, self.yn):
            self.yi = yi
            self.xi = 0
            self.first()
            path += f"M {self.p0l()},{self.p0w()} "
            for xi in range(self.xn):
                self.xi = xi
                path += (
                    f"C {self.p1l()} {self.p1w()} {self.p2l()} {self.p2w()} {self.p3l()} {self.p3w()} "
                    f"C {self.p4l()} {self.p4w()} {self.p5l()} {self.p5w()} {self.p6l()} {self.p6w()} "
                    f"C {self.p7l()} {self.p7w()} {self.p8l()} {self.p8w()} {self.p9l()} {self.p9w()} "
                )
                self.next()
        return path

    # --- Generate vertical puzzle paths ---
    def gen_dv(self):
        self.vertical = True
        path = ""
        for xi in range(1, self.xn):
            self.xi = xi
            self.yi = 0
            self.first()
            path += f"M {self.p0w()},{self.p0l()} "
            for yi in range(self.yn):
                self.yi = yi
                path += (
                    f"C {self.p1w()} {self.p1l()} {self.p2w()} {self.p2l()} {self.p3w()} {self.p3l()} "
                    f"C {self.p4w()} {self.p4l()} {self.p5w()} {self.p5l()} {self.p6w()} {self.p6l()} "
                    f"C {self.p7w()} {self.p7l()} {self.p8w()} {self.p8l()} {self.p9w()} {self.p9l()} "
                )
                self.next()
        return path

    # --- Generate outer border (with rounded corners) ---
    def gen_db(self):
        return (
            f"M {self.offset + self.radius} {self.offset} "
            f"L {self.offset + self.width - self.radius} {self.offset} "
            f"A {self.radius} {self.radius} 0 0 1 {self.offset + self.width} {self.offset + self.radius} "
            f"L {self.offset + self.width} {self.offset + self.height - self.radius} "
            f"A {self.radius} {self.radius} 0 0 1 {self.offset + self.width - self.radius} {self.offset + self.height} "
            f"L {self.offset + self.radius} {self.offset + self.height} "
            f"A {self.radius} {self.radius} 0 0 1 {self.offset} {self.offset + self.height - self.radius} "
            f"L {self.offset} {self.offset + self.radius} "
            f"A {self.radius} {self.radius} 0 0 1 {self.offset + self.radius} {self.offset} "
        )

    # --- Generate the final SVG string ---
    def generate_svg(self, embed_image_data=None):
        # For final output we use offset=0.
        self.offset = 0.0

        svg_parts = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.0" '
            f'width="{self.width}mm" height="{self.height}mm" viewBox="0 0 {self.width} {self.height}">'
        )
        # White background.
        svg_parts.append(f'<rect width="{self.width}" height="{self.height}" fill="white" />')
        # Optionally embed an image.
        if embed_image_data:
            svg_parts.append(
                f'<image x="0" y="0" width="{self.width}" height="{self.height}" preserveAspectRatio="none" href="data:image/png;base64,{embed_image_data}"/>'
            )
        # Add puzzle paths (all drawn with black strokes).
        svg_parts.append(f'<path fill="none" stroke="black" stroke-width="0.2" d="{self.gen_dh()}"></path>')
        svg_parts.append(f'<path fill="none" stroke="black" stroke-width="0.2" d="{self.gen_dv()}"></path>')
        svg_parts.append(f'<path fill="none" stroke="black" stroke-width="0.2" d="{self.gen_db()}"></path>')
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

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
    from the original image. The extracted pieces are saved in output_folder.
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
    
        piece_mask = np.zeros_like(template)
        cv2.drawContours(piece_mask, [cnt], -1, 255, thickness=-1)
        piece = cv2.bitwise_and(original, original, mask=piece_mask)
        x, y, w, h = cv2.boundingRect(cnt)
        piece_cropped = piece[y:y+h, x:x+w]
        out_path = os.path.join(output_folder, f"piece_{idx+1}.png")
        cv2.imwrite(out_path, piece_cropped)
        print(f"Saved piece {idx+1} to {out_path}")

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
                            xn = args.xn, yn = args.yn, threshold=args.threshold, dilate_edges=not args.no_dilate)
    
# Example run #
# python3 full_pipeline.py original.png --xn 5 --yn 5 --seed 1234 --use_original_size
# python3 src/full_pipeline.py data/img_2.jpg --xn 5 --yn 5 --seed 1234 --output_pieces_folder pieces_img_2 --use_original_size


if __name__ == "__main__":
    main()