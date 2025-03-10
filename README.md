# Jigsaw Puzzle Generator

This project takes an image file and divides it into jigsaw puzzle pieces, saving each piece as an SVG file.

## Project Structure

```
.
├── data/               # Place your input images here
├── output/             # Generated puzzle pieces will be saved here
├── src/                # Source code
│   ├── puzzlegen.py    # Core puzzle generation code
│   ├── clean_svgs.py   # Standalone SVG cleaning script 
│   ├── example.py      # Example script
│   └── run_puzzle_generator.py  # Main runner script with integrated cleaning
└── requirements.txt    # Dependencies
```

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: If you encounter issues with Inkscape, you may need to install it separately:
- On macOS: `brew install inkscape`
- On Ubuntu/Debian: `sudo apt-get install inkscape`
- On Windows: Download and install from [Inkscape's website](https://inkscape.org/release/)

2. Place your image files in the `data/` directory.

## Running the Puzzle Generator

### Generate and Clean SVGs in One Step (Recommended)

The generator now automatically removes the background rectangle from each puzzle piece:

```bash
# Use an image from the data directory (automatically picks the first one)
python src/run_puzzle_generator.py

# Show detailed cleaning information
python src/run_puzzle_generator.py --verbose

# Process all images in the data directory
python src/run_puzzle_generator.py --all

# Skip the cleaning step if you want the original SVGs
python src/run_puzzle_generator.py --no-clean
```

### Clean Existing SVGs Only

If you just want to clean existing SVG files without generating new ones:

```bash
# Clean all SVGs in the output directory
python src/run_puzzle_generator.py --clean-only

# With verbose output
python src/run_puzzle_generator.py --clean-only --verbose
```

### Options

- `--output-dir`: Directory to save puzzle pieces (default: "output")
- `--rows`: Number of rows in the puzzle (default: 5)
- `--cols`: Number of columns in the puzzle (default: 8)
- `--stroke-color`: Color of the puzzle piece outlines (default: "red")
- `--fill-color`: Fill color for the puzzle pieces (default: "black")
- `--no-clean`: Skip cleaning SVGs (keeps the background rectangle)
- `--clean-only`: Only clean existing SVGs without generating new ones
- `--verbose`: Show detailed information during the cleaning process
- `--all`: Process all images in the data directory
- `--list`: List all available images in the data directory

## What the SVG Cleaning Does

The cleaning process simply removes the background rectangle from each SVG file. This helps display each puzzle piece without the white rectangular background.

The background removal is intentionally minimal to avoid changing the appearance or positioning of the puzzle pieces themselves.

## Upload Images

Place your JPG or PNG image files in the `data/` directory. The script will automatically find them.

When processing multiple images, each image will:
1. Have its own subdirectory in the output folder
2. Be processed with the same rows/columns settings
3. Be named according to the original filename 