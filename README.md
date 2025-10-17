Overview
Welcome to the Random Image Puzzle Challenge from UCU-CSEA's Code Challenge Season 1! This project challenges you to reconstruct a scrambled image by solving two core problems: reordering fragmented pieces and correcting their rotations. It's a fun exercise in array manipulation, sorting, and image processing using Python and NumPy.
Challenge Goals

Reconstruct the Original Image: Start with a scrambled set of 81 image pieces (from a 450x450 pixel image divided into a 9x9 grid). Each piece is rotated randomly (0°, 90°, 180°, or 270°) and shuffled.
Key Skills Tested: Sorting algorithms, matrix rotations, NumPy array handling, and image reconstruction.

The puzzle simulates a three-step scrambling process:

Breakdown & Assign: Image split into 81 pieces, each tagged with a unique index k (1-81) for its original grid position (row-major order).
Transform: Random rotation applied to each piece's RGB matrix, with the angle θ stored.
Shuffle: Pieces randomly reordered into A_puzzle, a list of tuples: (rotated_matrix, k, θ).

Your task: Sort by k and undo rotations to rebuild the image.
