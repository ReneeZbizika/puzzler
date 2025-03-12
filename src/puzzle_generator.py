import math
import cairosvg

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
