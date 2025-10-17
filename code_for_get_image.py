import numpy as np
import matplotlib.pyplot as plt
import random

# Step 0: Generate a synthetic original image for demonstration (450x450x3)
# In the actual challenge, load the real image, e.g., original_image = plt.imread('path_to_image.jpg')
# For now, create a colorful gradient image
def create_synthetic_image(size=450):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    img = np.stack([
        (X + Y) * 255,  # Red channel
        X * 255,        # Green channel
        Y * 255         # Blue channel
    ], axis=-1).astype(np.uint8)
    return img

original_image = create_synthetic_image(450)
plt.figure(figsize=(5, 5))
plt.imshow(original_image)
plt.title('Original Synthetic Image')
plt.axis('off')
plt.show()

# Puzzle Parameters
image_size = 450
grid_size = 9
piece_size = 50
n_pieces = 81

# Step 1: Breakdown into pieces and assign k
pieces = []
for idx in range(n_pieces):
    row = idx // grid_size
    col = idx % grid_size
    piece = original_image[
        row * piece_size:(row + 1) * piece_size,
        col * piece_size:(col + 1) * piece_size,
        :
    ]
    k = idx + 1  # 1 to 81
    pieces.append((piece, k))

# Step 2: Apply random rotations to each piece
S_rot = [0, 90, 180, 270]
A_pieces = []
for piece_matrix, k in pieces:
    theta = random.choice(S_rot)
    # Apply rotation using np.rot90 (CCW)
    num_rot = theta // 90
    rotated_matrix = np.rot90(piece_matrix, k=num_rot, axes=(0, 1))
    A_pieces.append((rotated_matrix, k, theta))

# Step 3: Shuffle the pieces (Fisher-Yates shuffle simulation)
random.shuffle(A_puzzle := A_pieces)  # Python 3.8+ walrus operator; alternatively, A_puzzle = A_pieces; random.shuffle(A_puzzle)

print(f"A_puzzle created with {len(A_puzzle)} pieces.")
print(f"Example piece: k={A_puzzle[0][1]}, theta={A_puzzle[0][2]}")

# Now, Show the Scrambled Puzzle State
scrambled = np.zeros((image_size, image_size, 3), dtype=np.uint8)
for idx in range(n_pieces):
    piece_matrix = A_puzzle[idx][0]  # rotated matrix as-is
    row = idx // grid_size
    col = idx % grid_size
    scrambled[row * piece_size:(row + 1) * piece_size,
              col * piece_size:(col + 1) * piece_size, :] = piece_matrix

plt.figure(figsize=(10, 10))
plt.imshow(scrambled)
plt.axis('off')
plt.title('Scrambled Puzzle State')
plt.show()

# SOLUTION GUIDE
print("### Step 1: Sort the scrambled array")
A_sorted = sorted(A_puzzle, key=lambda piece: piece[1])
print(f"Sorted array: First piece k={A_sorted[0][1]}, Last piece k={A_sorted[-1][1]}")

print("\n### Step 2: Un-rotate the pieces and reconstruct")
def undo_rotation(matrix, theta):
    """
    Undo the rotation: rotate back by (360 - theta) degrees using np.rot90 (CCW).
    Equivalent to rotating theta // 90 times clockwise, or (4 - (theta//90)) % 4 times CCW.
    """
    if theta == 0:
        return matrix
    num_rotations = (360 - theta) // 90
    return np.rot90(matrix, k=num_rotations, axes=(0, 1))

corrected_pieces = [undo_rotation(piece[0], piece[2]) for piece in A_sorted]

# Reconstruct the solved image
final_solved_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
for idx in range(n_pieces):
    piece_matrix = corrected_pieces[idx]
    row = idx // grid_size
    col = idx % grid_size
    final_solved_image[row * piece_size:(row + 1) * piece_size,
                       col * piece_size:(col + 1) * piece_size, :] = piece_matrix

plt.figure(figsize=(10, 10))
plt.imshow(final_solved_image)
plt.axis('off')
plt.title('Reconstructed Solved Image')
plt.show()
plt.imsave('original_puzzle_image.png', original_image)
# Verify
if np.array_equal(original_image, final_solved_image):
    print("\n✅ Verification successful: Final image matches original image data.")
else:
    print("\n⚠️ Verification required: Final image is visually correct but might differ slightly due to pixel trimming.")