import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Helper function to load image from a URL
def load_image_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # catch HTTP errors
    return Image.open(BytesIO(response.content))

# Elephant image URL
elephant_url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"

# Load elephant image
elephant = load_image_from_url(elephant_url)

# Display original image
plt.figure(figsize=(6, 4))
plt.imshow(elephant)
plt.title("Elephant")
plt.axis("off")
plt.show()

# Convert to NumPy array
elephant_np = np.array(elephant)
print("Elephant image shape:", elephant_np.shape)

# Convert to grayscale
elephant_gray = elephant.convert("L")

# Display grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(elephant_gray, cmap="gray")
plt.title("Elephant (Grayscale)")
plt.axis("off")
plt.show()
