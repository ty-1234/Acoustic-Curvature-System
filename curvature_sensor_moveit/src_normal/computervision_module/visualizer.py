import cv2
import numpy as np
import signal
import sys

def visualize_results(image, gray, edges, points, curvatures):
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Closing windows and exiting...")
        cv2.destroyAllWindows()
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create a copy of the original image to draw on
    output = image.copy()
    
    # Iterate through points and their corresponding curvatures
    for pt, curv in zip(points, curvatures):
        x, y = pt  # Extract x and y coordinates of the point
        # Calculate intensity based on curvature, scaled to 0-255
        intensity = int(min(curv * 255 / np.pi, 255))
        # Draw a circle at the point with color based on curvature intensity
        cv2.circle(output, (x, y), 2, (0, intensity, 255 - intensity), -1)

    # Display the original image
    cv2.imshow("Original", image)
    # Display the grayscale version of the image
    cv2.imshow("Grayscale", gray)
    # Display the edges detected in the image
    cv2.imshow("Edges", edges)
    # Display the image with curvature estimation visualization
    cv2.imshow("Curvature Estimation", output)
    
    print("Press any key in the visualization windows to close them")
    print("Or press Ctrl+C in the terminal to exit")
    
    # Wait for a key press to close the windows, but check more frequently
    while True:
        # Check for key press every 100ms instead of waiting indefinitely
        if cv2.waitKey(100) != -1:
            break
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()