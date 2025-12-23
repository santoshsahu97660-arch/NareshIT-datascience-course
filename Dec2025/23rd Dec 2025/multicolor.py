import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Safe image loader
def load_image_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # Check request success
    if response.status_code != 200:
        raise Exception("❌ Failed to download image")

    # Check content type
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise Exception("❌ URL did not return an image")

    return Image.open(BytesIO(response.content)).convert("RGB")


# Elephant image URL
elephant_url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"

# Load image
elephant = load_image_from_url(elephant_url)
elephant_np = np.array(elephant)

# Split RGB channels
R, G, B = elephant_np[:, :, 0], elephant_np[:, :, 1], elephant_np[:, :, 2]

# Create channel emphasized images
red_img = np.zeros_like(elephant_np)
green_img = np.zeros_like(elephant_np)
blue_img = np.zeros_like(elephant_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display original and RGB images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(elephant_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(red_img)
plt.title("Red Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(green_img)
plt.title("Green Channel Emphasis")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blue_img)
plt.title("Blue Channel Emphasis")
plt.axis("off")

plt.tight_layout()
plt.show()

# Grayscale + Colormap
elephant_gray = elephant.convert("L")
elephant_gray_np = np.array(elephant_gray)

plt.figure(figsize=(6, 5))
plt.imshow(elephant_gray_np, cmap="viridis")
plt.title("Colormapped Grayscale")
plt.axis("off")
plt.colorbar()
plt.show()
