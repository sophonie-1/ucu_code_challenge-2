# 🧩 Random Image Puzzle Challenge

# 🎓 University Coding Challenge — UCU Code Challenge Season 1

**Goal:** Reconstruct a scrambled image by correctly reordering and reorienting its fragmented pieces using Python.

---

## 📘 Introduction

This project demonstrates how to solve the **Random Image Puzzle Challenge**, where an original image is divided, rotated, and shuffled.
Your task as a solver is to use Python to restore the original image.

---

# 🧠 Problem Overview

The challenge begins with an image that has been:

1. **Fragmented** into 81 small pieces (forming a 9×9 grid).
2. **Randomly rotated** by one of four angles: 0°, 90°, 180°, or 270°.
3. **Shuffled** in a random order.

Each piece is represented as:

```
Piece_i = (Rotated RGB Matrix, Correct Order Index k, Rotation Angle θ)
```

# Your mission:

1. **Sort** all the pieces by their correct index `k`.
2. **Undo** the rotation `θ` applied to each piece.
3. **Reconstruct** the full 450×450 image by joining the corrected pieces.

---

## 🧰 Technologies Used

| Library          | Purpose                                            |
| ---------------- | -------------------------------------------------- |
| **numpy**        | For handling image matrices and reshaping data     |
| **Pillow (PIL)** | For loading, rotating, and processing image pieces |
| **matplotlib**   | For visualizing the final reconstructed image      |
| **random**       | To simulate and shuffle puzzle pieces              |

---

# ⚙️ Installation

Clone this repository or open the notebook in **Google Colab**.

# 🪄 Option 1 — Run on Google Colab

Simply open the notebook from the challenge link:
[Google Colab Puzzle Notebook](https://colab.research.google.com/github/UCU-CSEA/UCU-Code-Challenge-Season-1/blob/main/puzzle.ipynb)

# 💻 Option 2 — Run Locally

If you want to run it on your computer:

```bash
pip install numpy pillow matplotlib
```

Then open the notebook:

```bash
jupyter notebook puzzle.ipynb
```

---

# 🧩 How It Works

1. **Load puzzle data** (scrambled image pieces).
2. **Sort by index (`k`)** to restore the original sequence.
3. **Undo rotation (`θ`)** using Pillow’s `rotate()` function.
4. **Rebuild the full grid** by merging pieces row by row using NumPy.
5. **Display final image** using `matplotlib`.

---

# 🪜 Steps to Reconstruct the Image

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random, math

# Sort pieces
pieces_sorted = sorted(A_puzzle, key=lambda x: x[1])

# Undo rotation
def undo_rotation(image, angle):
    return np.array(Image.fromarray(image.astype(np.uint8)).rotate(-angle))

pieces_corrected = [(undo_rotation(img, rot), k, rot) for (img, k, rot) in pieces_sorted]

# Rebuild grid
grid_size = int(math.sqrt(len(pieces_corrected)))
rows = []
for i in range(grid_size):
    row = np.concatenate([pieces_corrected[j][0] for j in range(i*grid_size, (i+1)*grid_size)], axis=1)
    rows.append(row)

final_image = np.concatenate(rows, axis=0)
plt.imshow(final_image.astype(np.uint8))
plt.axis('off')
plt.show()
```
# To create image file

````run in terminal or in vs code```
----
python code_for_get_image.py
---
but you need to delete that first image to avoid name confusion renaming


# 📊 Example Output

After executing all steps, the scrambled image is successfully reconstructed to its original form.

---

# 🧑‍💻 Contributors

* **Bukira Sophonie** — Developer & Solver
* **UCU Code Challenge Organizers** — Puzzle creators

---

# 🏁 License

This project is for educational purposes under the **UCU Code Challenge Season 1**.
You may reuse or modify the code with attribution.
