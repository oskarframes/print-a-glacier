#!/usr/bin/env python3

"""
Convert a grayscale heightmap to a watertight STL.

Requirements:
    pip install numpy pillow numpy-stl

White pixels -> high elevation
Black pixels -> low elevation

The script:
    - Crops the image
    - Downsamples to a manageable resolution
    - Generates a watertight STL with a flat base
"""

import numpy as np
from PIL import Image
from stl import mesh

# ==========================================================
# INPUT / OUTPUT
# ==========================================================

IMAGE_FILE = "10k.jpg"
OUTPUT_FILE = "terrain.stl"

# ==========================================================
# CROP SETTINGS
# ==========================================================

CROP_TOP = 0.30
CROP_BOTTOM = 0.20
CROP_LEFT = 0.01
CROP_RIGHT = 0.20

# ==========================================================
# MODEL SETTINGS
# ==========================================================

# Maximum resolution after downsampling
TARGET_RESOLUTION = 1500

# Physical size (mm)
WIDTH_MM = 200

# Height exaggeration
MAX_HEIGHT_MM = 30

# Flat base thickness
BASE_THICKNESS_MM = 3

# Flip vertically (north stays north)
FLIP_Y = True

# ==========================================================
# LOAD IMAGE
# ==========================================================

print("Loading image...")

img = Image.open(IMAGE_FILE).convert("L")

w, h = img.size

left = int(w * CROP_LEFT)
right = int(w * (1.0 - CROP_RIGHT))
top = int(h * CROP_TOP)
bottom = int(h * (1.0 - CROP_BOTTOM))

img = img.crop((left, top, right, bottom))

print(f"Cropped size: {img.size}")

# Downsample
img.thumbnail((TARGET_RESOLUTION, TARGET_RESOLUTION),
              Image.Resampling.LANCZOS)

print(f"Working resolution: {img.size}")

heightmap = np.asarray(img, dtype=np.float32) / 255.0

if FLIP_Y:
    heightmap = np.flipud(heightmap)

rows, cols = heightmap.shape

# Preserve aspect ratio
HEIGHT_MM = WIDTH_MM * rows / cols

dx = WIDTH_MM / (cols - 1)
dy = HEIGHT_MM / (rows - 1)

# ==========================================================
# CREATE VERTICES
# ==========================================================

heightmap = np.sqrt(heightmap)
print("Generating vertices...")

vertices = []

# Top
for y in range(rows):
    for x in range(cols):
        z = BASE_THICKNESS_MM + heightmap[y, x] * MAX_HEIGHT_MM
        vertices.append([x * dx, y * dy, z])

bottom_offset = len(vertices)

# Bottom
for y in range(rows):
    for x in range(cols):
        vertices.append([x * dx, y * dy, 0])

vertices = np.array(vertices)


def vid(x, y):
    return y * cols + x


# ==========================================================
# CREATE FACES
# ==========================================================

print("Generating mesh...")

faces = []

# ---------------- Top ----------------

for y in range(rows - 1):
    for x in range(cols - 1):

        v0 = vid(x, y)
        v1 = vid(x + 1, y)
        v2 = vid(x, y + 1)
        v3 = vid(x + 1, y + 1)

        faces.append([v0, v2, v1])
        faces.append([v1, v2, v3])

# ---------------- Bottom ----------------

for y in range(rows - 1):
    for x in range(cols - 1):

        v0 = bottom_offset + vid(x, y)
        v1 = bottom_offset + vid(x + 1, y)
        v2 = bottom_offset + vid(x, y + 1)
        v3 = bottom_offset + vid(x + 1, y + 1)

        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])

# ---------------- Left wall ----------------

for y in range(rows - 1):

    t0 = vid(0, y)
    t1 = vid(0, y + 1)

    b0 = bottom_offset + t0
    b1 = bottom_offset + t1

    faces.append([t0, b0, t1])
    faces.append([t1, b0, b1])

# ---------------- Right wall ----------------

for y in range(rows - 1):

    t0 = vid(cols - 1, y)
    t1 = vid(cols - 1, y + 1)

    b0 = bottom_offset + t0
    b1 = bottom_offset + t1

    faces.append([t0, t1, b0])
    faces.append([t1, b1, b0])

# ---------------- Bottom wall ----------------

for x in range(cols - 1):

    t0 = vid(x, 0)
    t1 = vid(x + 1, 0)

    b0 = bottom_offset + t0
    b1 = bottom_offset + t1

    faces.append([t0, t1, b0])
    faces.append([t1, b1, b0])

# ---------------- Top wall ----------------

for x in range(cols - 1):

    t0 = vid(x, rows - 1)
    t1 = vid(x + 1, rows - 1)

    b0 = bottom_offset + t0
    b1 = bottom_offset + t1

    faces.append([t0, b0, t1])
    faces.append([t1, b0, b1])

print(f"Vertices : {len(vertices):,}")
print(f"Triangles: {len(faces):,}")

# ==========================================================
# EXPORT STL
# ==========================================================

print("Writing STL...")

terrain = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

for i, face in enumerate(faces):
    terrain.vectors[i] = vertices[face]

terrain.save(OUTPUT_FILE)

print()
print("Done!")
print(f"Saved to {OUTPUT_FILE}")