import cv2
import numpy as np

# Parameters for drawing
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates of the region

# List to store segmentation points
annotations = []

# Mouse callback function to draw contours
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        annotations.append([(x, y)])  # Start a new contour

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Add points to the current contour
            annotations[-1].append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Close the contour by connecting the last point to the first
        annotations[-1].append((x, y))

# Function to display the image and collect annotations
def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    # Create a clone of the image for annotation display
    annotated_image = image.copy()
    cv2.namedWindow("Image Segmentation")
    cv2.setMouseCallback("Image Segmentation", draw_contour)

    while True:
        # Show the annotations on the cloned image
        temp_image = annotated_image.copy()
        for contour in annotations:
            points = np.array(contour, dtype=np.int32)
            cv2.polylines(temp_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Convert the image to grayscale and apply edge detection
        gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edges image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the polygon has 4 points, it might be a rectangle (or square)
            if len(approx) == 4:
                # Get the bounding box (x, y, W, H)
                x, y, w, h = cv2.boundingRect(approx)

                # Check if it is a square (w == h)
                if abs(w - h) < 10:  # Allow small margin for error
                    cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle

                    # Return (x, y, h, w) for the square
                    print(f"Square Detected: (x, y, h, w) = ({x}, {y}, {h}, {w})")

        # Display the image with annotations and detected squares
        cv2.imshow("Image Segmentation", temp_image)
        
        # Press 's' to save annotations, 'c' to clear, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            with open("annotations.txt", "w") as f:
                for contour in annotations:
                    f.write(str(contour) + "\n")
            print("Annotations saved to annotations.txt")
        elif key == ord("c"):
            # Clear annotations
            annotations.clear()
            annotated_image = image.copy()
            print("Annotations clear")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\82104\OneDrive\문서\clone_pre\Image_dataset\000000000139.jpg"
    print("Image path:", image_path)  # 경로 확인용 출력
    segment_image(image_path)

