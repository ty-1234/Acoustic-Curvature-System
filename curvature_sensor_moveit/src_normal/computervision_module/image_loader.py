import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Load the image from the specified file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image.

    Raises:
        FileNotFoundError: If the image file is not found.
        ValueError: If the image format is unsupported.
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image '{image_path}' not found.")

    # Explicitly reject unsupported formats like .heic
    if image_path.lower().endswith(".heic"):
        raise ValueError("Error: .heic files are not supported. Please use .jpg or .png files.")

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unsupported image format for '{image_path}'.")
    return image

def preprocess_image(image):
    """Converts image to grayscale, ensures it's CV_8U, and applies Gaussian blur."""
    if image is None:
        print("preprocess_image: Input image is None.")
        return None

    if not isinstance(image, np.ndarray):
        print(f"preprocess_image: Input is not a NumPy array. Type: {type(image)}")
        return None

    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2: # Already single channel
        gray = image.copy() 
    else:
        print(f"preprocess_image: Unsupported image shape {image.shape}. Expected 2D or 3D with 3 channels.")
        return None

    # Ensure the image is 8-bit unsigned (CV_8U)
    if gray.dtype != np.uint8:
        print(f"preprocess_image: Grayscale image dtype is {gray.dtype}, not uint8. Converting.")
        if np.issubdtype(gray.dtype, np.floating): # If float, assume range [0,1] or [0,255]
            if gray.min() >= 0 and gray.max() <= 1: # Likely [0,1] range
                 gray = (gray * 255.0).astype(np.uint8)
            else: # Assume it's float but in [0,255] range already or needs clipping
                 gray = np.clip(gray, 0, 255).astype(np.uint8)
        elif np.issubdtype(gray.dtype, np.integer):
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        else:
            print(f"preprocess_image: Unhandled dtype {gray.dtype} for conversion to uint8.")
            return None
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred