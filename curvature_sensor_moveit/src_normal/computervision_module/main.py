import os
import cv2
import numpy as np
from image_loader import load_image, preprocess_image
from edge_detector import detect_edges
from contour_utils import find_largest_contour, estimate_curvature
from csv_writer import save_curvature_to_csv
from visualizer import visualize_results
from curvature_converter import convert_curvature_to_meters

def get_lasso_selection(image):
    """
    Allow the user to draw a free-form lasso selection around the region of interest.
    
    Args:
        image (numpy.ndarray): Input image to be displayed for selection.
        
    Returns:
        numpy.ndarray: Binary mask where the selected region is set to 255.
    """
    # Make a copy of the image to draw on
    drawing_image = image.copy()
    # Create a black mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # List to store the lasso points
    lasso_points = []
    # Flag to indicate if drawing is active
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, drawing_image, mask, lasso_points
        
        # Start drawing on left button down
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            lasso_points = [(x, y)]
            drawing_image = image.copy()
            
        # Continue drawing while the left button is pressed and moving
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(drawing_image, lasso_points[-1], (x, y), (0, 255, 0), 2)
            lasso_points.append((x, y))
            
        # End drawing on left button up
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Close the lasso by connecting the last point to the first point
            if len(lasso_points) > 2:
                cv2.line(drawing_image, lasso_points[-1], lasso_points[0], (0, 255, 0), 2)
                # Convert lasso points to numpy array and draw filled polygon on mask
                points = np.array(lasso_points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
    
    # Create a window and set the mouse callback
    window_name = "Draw Lasso - Draw around your region of interest and press 'Enter' when done"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "="*50)
    print("Instructions for Lasso Selection:")
    print("- Hold down the left mouse button and draw around your region of interest.")
    print("- Release the mouse button when you're done drawing.")
    print("- Press ENTER to confirm the selection.")
    print("- Press ESC to cancel and try again.")
    print("="*50)
    
    while True:
        cv2.imshow(window_name, drawing_image)
        # Wait for key press (10ms delay)
        key = cv2.waitKey(10)
        
        # Press Enter to confirm the selection
        if key == 13:  # Enter key
            break
        # Press Esc to reset the selection
        elif key == 27:  # Esc key
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            drawing_image = image.copy()
            lasso_points = []
    
    cv2.destroyWindow(window_name)
    return mask

def main():
    """
    Main entry point of the program. It orchestrates the workflow:
    1. Load and preprocess the image.
    2. Allow the user to select a region of interest (ROI) using a lasso tool.
    3. Detect edges and find the largest contour in the ROI.
    4. Estimate curvature and save results to a CSV file.
    5. Visualize the results.
    """
    # Locate the computervision_module directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_folder = os.path.join(base_dir, 'test_image')

    # Ensure the test_image folder exists
    if not os.path.exists(test_image_folder):
        raise FileNotFoundError(f"The folder 'test_image' was not found in {base_dir}. Please create it and add images.")

    # List all available images in the test_image folder
    print("\n" + "="*50)
    print("Available images in 'test_image' folder:")
    available_images = [f for f in os.listdir(test_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_images:
        raise FileNotFoundError(f"No images found in the folder: {test_image_folder}")

    # Display the list of images with corresponding numbers
    for idx, img_name in enumerate(available_images, start=1):
        print(f"{idx}. {img_name}")
    print("="*50)

    # Prompt the user to select an image
    while True:
        try:
            image_choice = int(input("Enter the number corresponding to the image you want to process: "))
            if 1 <= image_choice <= len(available_images):
                image_name = available_images[image_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_images)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Construct the full path to the selected image
    image_path = os.path.join(test_image_folder, image_name)
    print(f"Selected image: {image_name}")

    # Define camera parameters TODO: Update these values based on your camera setup in franka Lab
    camera_height_m = (25.5-5)/100  # Camera height from the table in meters
    focal_length_mm = 1.93  # Camera focal length in mm
    sensor_width_mm = 3.6  # Camera sensor width in mm

    # Extract the image name (without extension) for saving results
    image_name_no_ext = os.path.splitext(image_name)[0]

    # Step 1: Load the image
    image = load_image(image_path)
    
    # Step 2: Preprocess the full image (convert to grayscale and apply Gaussian blur)
    full_gray, full_blurred = preprocess_image(image)
    
    # Step 3: Detect edges in the preprocessed full image
    full_edges = detect_edges(full_blurred)
    
    # Step 4: Allow the user to select a region of interest (ROI) using the lasso tool
    mask = get_lasso_selection(image)
    
    # Get the bounding box of the mask for visualization purposes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        raise ValueError("No region selected. Please try again.")
    
    # Apply the mask to the edge image
    roi_edges = cv2.bitwise_and(full_edges, mask)
    
    # Step 5: Find contours in the ROI of the edge-detected image
    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filter contours to only include those within the ROI
    roi_contours = []
    for contour in contours:
        # Check if contour is mostly within the masked area
        contour_mask = np.zeros_like(full_edges)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        roi_area = cv2.countNonZero(cv2.bitwise_and(contour_mask, mask))
        contour_area = cv2.countNonZero(contour_mask)
        if contour_area > 0 and roi_area / contour_area > 0.5:  # If more than 50% of contour is in ROI
            roi_contours.append(contour)
    
    # Find the largest contour among the filtered ones
    if roi_contours:
        contour = max(roi_contours, key=cv2.contourArea)
    else:
        raise ValueError("No contours found in the selected region. Try selecting a different area.")

    # Step 6: Estimate the curvature of the largest contour
    # Try a larger window, e.g., 15, 20, 25, 30 or more.
    # The optimal value will depend on the scale of your object in the image and contour point density.
    points, curvatures = estimate_curvature(contour, window=20) # Example: changed to 20

    # Step 7: Convert curvature from pixels^-1 to meters^-1
    image_width_px = image.shape[1]
    curvatures_meters, pixels_per_meter = convert_curvature_to_meters(
        curvatures, camera_height_m, focal_length_mm, sensor_width_mm, image_width_px
    )
    print(f"Conversion factor (pixels_per_meter): {pixels_per_meter:.4f}")

    # Step 8: Save the curvature data (points and curvatures) to a CSV file
    csv_path = save_curvature_to_csv(
        points, 
        curvatures_meters, 
        image_name_no_ext, 
        camera_height_m, 
        focal_length_mm, 
        sensor_width_mm, 
        pixels_per_meter
    )
    
    # Step 9: Print curvature statistics (mean and maximum curvature)
    print("\n" + "="*50)
    print("Processing Results:")
    print(f"Curvature data saved to: {csv_path}")
    print(f"Mean Curvature (meters^-1): {np.mean(curvatures_meters):.4f}")
    print(f"Max Curvature (meters^-1): {np.max(curvatures_meters):.4f}")
    print(f"Mean Curvature (pixels^-1): {np.mean(curvatures):.4f}")
    print(f"Max Curvature (pixels^-1): {np.max(curvatures):.4f}")
    print("="*50)

    # Step 10: Create a visualization image
    # Draw the selected contour on the original image
    viz_image = image.copy()
    cv2.drawContours(viz_image, [contour], 0, (0, 255, 0), 2)
    
    # Draw the lasso selection area
    lasso_viz = np.zeros_like(viz_image)
    lasso_viz[mask > 0] = [0, 0, 255]  # Red color for selected area
    viz_image = cv2.addWeighted(viz_image, 1, lasso_viz, 0.3, 0)
    
    # Extract ROI for visualization (using bounding box)
    roi_image = image[y:y+h, x:x+w].copy()
    roi_gray = full_gray[y:y+h, x:x+w]
    roi_edges_vis = full_edges[y:y+h, x:x+w]
    
    # Create a version of the mask cropped to the bounding box
    roi_mask = mask[y:y+h, x:x+w]
    
    # Adjust points coordinates for ROI visualization
    roi_points = [(px-x, py-y) for px, py in points]

    # Step 11: Visualize the results
    # Apply the mask to the cropped region for visualization
    roi_edges_masked = cv2.bitwise_and(roi_edges_vis, roi_edges_vis, mask=roi_mask)
    visualize_results(roi_image, roi_gray, roi_edges_masked, roi_points, curvatures_meters)
    
    # Show the full image with contour and lasso selection
    cv2.imshow("Full Image Analysis", viz_image)
    cv2.waitKey(100)  # Just a short delay to let the window appear

if __name__ == "__main__":
    """
    Entry point of the script.
    """
    main()