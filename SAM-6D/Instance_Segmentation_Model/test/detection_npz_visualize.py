import cv2
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file.npz' with the path to your .npz file
npz_data = np.load("detection_ism_0700.npz")

# Extract relevant data
masks = npz_data["segmentation"]
scores = npz_data["score"]
bboxes = npz_data["bbox"]

num_det, H, W = masks.shape
# Create a base image to overlay masks
canvas = np.zeros((H, W, 3), dtype=np.uint8)

# Randomly generate colors for each mask
colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(masks))]

# Overlay each mask and add bounding boxes with scores
for i, mask in enumerate(masks):
    color = colors[i]
    # Normalize mask and convert to 8-bit for cv2
    mask_normalized = (mask * 255).astype(np.uint8)
    # Create an overlay with the mask
    overlay = np.zeros_like(canvas)
    overlay[mask_normalized > 128] = color  # Apply color where mask is active

    # Blend the overlay onto the canvas
    canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.5, 0)

    # Draw bounding box
    x, y, w, h = bboxes[i]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

    # Add confidence score as text
    cv2.putText(
        canvas,
        f"{scores[i]:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

# Display the resulting image
cv2.imshow("Segmentation Masks with Scores", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
