import os
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from image_loader import load_image # Assuming preprocess_image is not strictly needed for point clicking
from contour_utils import estimate_curvature 
from csv_writer import save_curvature_to_csv
from curvature_converter import convert_curvature_to_meters

# get_lasso_selection can be kept if you want it as an alternative, or removed.
# For this request, we are focusing on the point clicking and spline generation.
# ... (get_lasso_selection function code as before) ...
def get_lasso_selection(image):
    """
    Allow the user to draw a free-form lasso selection around the region of interest.
    (This function can be kept for alternative use or removed if get_polyline_from_clicks is preferred)
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
                points_arr = np.array(lasso_points, dtype=np.int32) # Renamed to avoid conflict
                cv2.fillPoly(mask, [points_arr], 255)
    
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


def get_polyline_from_clicks(image):
    """
    Allows the user to define a polyline by clicking points on the image.
    Press ENTER to finalize the polyline. Press ESC to clear and restart.
    Press 'c' to cancel selection entirely.
    Args:
        image (numpy.ndarray): Input image to be displayed for point selection.
    Returns:
        numpy.ndarray: An array of points representing the polyline,
                       in the format suitable for OpenCV contours (N, 1, 2), or None if cancelled/insufficient.
    """
    clicked_points = [] 
    drawing_image = image.copy()
    window_name = "Click minimum of 4 points to define curve (ENTER to finish, ESC to clear, 'c' to cancel)"
    cv2.namedWindow(window_name)

    def mouse_callback_polyline(event, x, y, flags, param):
        nonlocal clicked_points, drawing_image
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            cv2.circle(drawing_image, (x, y), 3, (0, 255, 0), -1)
            if len(clicked_points) > 1:
                cv2.line(drawing_image, clicked_points[-2], clicked_points[-1], (0, 255, 0), 2)
            cv2.imshow(window_name, drawing_image)

    cv2.setMouseCallback(window_name, mouse_callback_polyline)
    cv2.imshow(window_name, drawing_image)

    print("\n" + "="*50)
    print("Instructions for Polyline Selection:")
    print("- Click points along the curve you want to analyze.")
    print("- Lines will connect the points as you click.")
    print("- Press ENTER to confirm the polyline.")
    print("- Press ESC to clear all points and start over.")
    print("- Press 'c' to close and cancel selection.")
    print("="*50)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER key
            break
        elif key == 27:  # ESC key
            clicked_points = []
            drawing_image = image.copy()
            cv2.imshow(window_name, drawing_image)
            print("Polyline cleared. Click again or press ENTER/c.")
        elif key == ord('c'):
            clicked_points = [] 
            print("Polyline selection cancelled.")
            break
    
    cv2.destroyWindow(window_name)

    # For spline fitting (cubic, k=3), we need at least k+1 = 4 points.
    # Menger curvature itself needs at least 3 points for the direct polyline.
    if not clicked_points or len(clicked_points) < 3: 
        print("Warning: Not enough points selected (need at least 3 for polyline, 4 for spline). No contour returned.")
        return None
    
    contour_array = np.array(clicked_points, dtype=np.int32).reshape((-1, 1, 2))
    return contour_array


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_folder = os.path.join(base_dir, 'test_image')

    if not os.path.exists(test_image_folder):
        raise FileNotFoundError(f"The folder 'test_image' was not found in {base_dir}. Please create it and add images.")

    print("\n" + "="*50)
    print("Available images in 'test_image' folder:")
    available_images = [f for f in os.listdir(test_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_images:
        raise FileNotFoundError(f"No images found in the folder: {test_image_folder}")

    for idx, img_name in enumerate(available_images, start=1):
        print(f"{idx}. {img_name}")
    print("="*50)

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

    image_path = os.path.join(test_image_folder, image_name)
    print(f"Selected image: {image_name}")

    camera_height_m = (25.5-5)/100
    focal_length_mm = 1.93
    sensor_width_mm = 3.6
    image_name_no_ext = os.path.splitext(image_name)[0]

    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    print("Please define the curve by clicking points on the image.")
    clicked_contour = get_polyline_from_clicks(image.copy()) 

    if clicked_contour is None:
        print("Contour definition failed or was cancelled. Exiting.")
        return

    contour_to_analyze = clicked_contour 
    fitted_spline_for_viz = None 
    use_spline_fitting = True 

    # --- Modified Spline Fitting Section ---
    TARGET_DENSE_SPLINE_POINTS = 150 # Fixed number of points for the final spline
    MIN_POINTS_FOR_SPLINE = 4 # Need k+1 points for cubic spline (k=3)

    if use_spline_fitting and len(clicked_contour) >= MIN_POINTS_FOR_SPLINE:
        try:
            print(f"Attempting to fit spline to {len(clicked_contour)} clicked points.")
            x_coords = clicked_contour[:, 0, 0]
            y_coords = clicked_contour[:, 0, 1]

            # Adaptive smoothing factor 's'
            num_clicked = len(x_coords)
            if num_clicked < 8: # For few points, interpolate
                smoothing_factor_s = 0
                print("Using interpolation (s=0) for spline fitting (few points).")
            else: # For more points, apply some smoothing
                  # Heuristic: s = m - sqrt(2m). Ensure s is not negative.
                smoothing_factor_s = max(0, num_clicked - np.sqrt(2 * num_clicked))
                print(f"Using adaptive smoothing (s={smoothing_factor_s:.2f}) for spline fitting.")

            tck, u = splprep([x_coords, y_coords], s=smoothing_factor_s, per=False, k=3)
            
            u_dense = np.linspace(u.min(), u.max(), TARGET_DENSE_SPLINE_POINTS)
            x_dense, y_dense = splev(u_dense, tck)

            smooth_contour_points = np.vstack((x_dense, y_dense)).T
            contour_to_analyze = smooth_contour_points.reshape((-1, 1, 2)).astype(np.int32)
            fitted_spline_for_viz = contour_to_analyze.copy()
            print(f"Spline fitted. Using {len(contour_to_analyze)} dense points for curvature analysis.")
        except Exception as e:
            print(f"Spline fitting failed: {e}. Using direct polyline from clicked points.")
            # contour_to_analyze remains clicked_contour
    elif use_spline_fitting:
        print(f"Not enough points ({len(clicked_contour)}) for spline fitting (need at least {MIN_POINTS_FOR_SPLINE}). Using direct polyline.")
    else:
        print("Spline fitting disabled. Using direct polyline from clicked points.")
    # --- End of Modified Spline Fitting Section ---
    
    # Adjust chosen_window based on the density of contour_to_analyze
    # If contour_to_analyze is the dense spline (e.g., 150 points):
    if len(contour_to_analyze) == TARGET_DENSE_SPLINE_POINTS and use_spline_fitting and fitted_spline_for_viz is not None:
        chosen_window = 7 # Example: 5-10% of half the number of points in a local segment.
                          # For 150 points, a window of 7 means looking ~1/10th of the curve length.
                          # This needs tuning based on visual results and accuracy.
        print(f"Using chosen_window = {chosen_window} for dense spline.")
    else: # If using direct polyline or spline fitting failed
        chosen_window = 1 # For sparse, user-clicked points, a small window is better.
        print(f"Using chosen_window = {chosen_window} for direct polyline.")


    if len(contour_to_analyze) < 2 * chosen_window + 1:
        print(f"Not enough points ({len(contour_to_analyze)}) in the contour for the chosen window size ({chosen_window}).")
        print(f"Please click more points or adjust parameters. Need at least {2 * chosen_window + 1} points.")
        return

    points_p2_for_curvature, curvatures_pixels = estimate_curvature(contour_to_analyze, window=chosen_window) 

    if not curvatures_pixels:
        print("Curvature estimation returned no results.")
        return

    image_width_px = image.shape[1]
    curvatures_meters, pixels_per_meter = convert_curvature_to_meters(
        curvatures_pixels, camera_height_m, focal_length_mm, sensor_width_mm, image_width_px
    )
    print(f"Conversion factor (pixels_per_meter): {pixels_per_meter:.4f}")

    csv_path = save_curvature_to_csv(
        points_p2_for_curvature, 
        curvatures_meters, 
        image_name_no_ext, 
        camera_height_m, 
        focal_length_mm, 
        sensor_width_mm, 
        pixels_per_meter
    )
    
    print("\n" + "="*50)
    print("Processing Results:")
    print(f"Curvature data saved to: {csv_path}")
    if curvatures_meters: 
        print(f"Mean Curvature (meters^-1): {np.mean(curvatures_meters):.4f}")
        print(f"Max Curvature (meters^-1): {np.max(curvatures_meters):.4f}")
    else:
        print("No metric curvature data to display.")
    if curvatures_pixels: 
        print(f"Mean Curvature (pixels^-1): {np.mean(curvatures_pixels):.4f}")
        print(f"Max Curvature (pixels^-1): {np.max(curvatures_pixels):.4f}")
    else:
        print("No pixel curvature data to display.")
    print("="*50)

    viz_image = image.copy()
    cv2.polylines(viz_image, [clicked_contour], isClosed=False, color=(0, 0, 255), thickness=1) 
    
    if fitted_spline_for_viz is not None:
         cv2.polylines(viz_image, [fitted_spline_for_viz], isClosed=False, color=(0, 255, 0), thickness=2) 
    elif use_spline_fitting == False or len(clicked_contour) < MIN_POINTS_FOR_SPLINE : # Draw clicked contour thicker if it's the one analyzed
         cv2.polylines(viz_image, [clicked_contour], isClosed=False, color=(0, 255, 0), thickness=2)


    for pt in points_p2_for_curvature:
        cv2.circle(viz_image, tuple(pt), 2, (255, 0, 0), -1) 

    cv2.imshow("User-Defined Curve Analysis", viz_image)
    print("\nPress any key in the visualization windows to close them.")
    print("Or press Ctrl+C in the terminal to exit.")
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key != 255: 
            break
        # Check if window was closed by clicking 'x'
        # Ensure the window name matches what's used in cv2.imshow
        if cv2.getWindowProperty("User-Defined Curve Analysis", cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("ImportError: Failed to import scipy.interpolate. Please ensure SciPy is installed.")
        print("You can install it using: pip install scipy")
    except Exception as e:
        print(f"An error occurred: {e}")
        # import traceback # For debugging
        # traceback.print_exc() # For debugging
    finally:
        cv2.destroyAllWindows() 
        print("Exiting program.")