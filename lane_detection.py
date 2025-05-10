import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform to detect edges"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Define a blank mask to start with
    mask = np.zeros_like(img)
    
    # Define a 3 channel or 1 channel color to fill the mask with
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Fill pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, 
                            maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def filter_white(img):
    """
    Filter the image to only show white-ish areas.
    This helps to isolate white lane markings.
    """
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for white color - more selective thresholds
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    
    # Create a mask that identifies white pixels
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Alternate method: use RGB threshold directly
    # Convert image to RGB (OpenCV uses BGR by default)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create thresholds for all channels to find white
    r_thresh = rgb[:,:,0] > 220
    g_thresh = rgb[:,:,1] > 220
    b_thresh = rgb[:,:,2] > 220
    
    # Combine thresholds
    rgb_white = np.zeros_like(white_mask)
    rgb_white[r_thresh & g_thresh & b_thresh] = 255
    
    # Combine both masks for better results
    combined_mask = cv2.bitwise_or(white_mask, rgb_white)
    
    # Enhance visibility with morphological operations
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    return combined_mask

def process_image(image):
    """
    Process an image to identify and draw lane lines.
    """
    # Step 1: Filter for white colors (lane markings)
    white_filtered = filter_white(image)
    
    # Step 2: Apply Gaussian blur to reduce noise
    kernel_size = 5
    blurred = gaussian_blur(white_filtered, kernel_size)
    
    # Step 3: Apply Canny edge detection
    low_threshold = 30
    high_threshold = 100
    edges = canny(blurred, low_threshold, high_threshold)
    
    # Step 4: Define region of interest
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), 
                          (imshape[1] * 0.4, imshape[0] * 0.6), 
                          (imshape[1] * 0.6, imshape[0] * 0.6), 
                          (imshape[1], imshape[0])]], 
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Step 5: Apply Hough transform to detect lines
    rho = 2
    theta = np.pi/180
    threshold = 20
    min_line_length = 30
    max_line_gap = 20
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    # Create a blank color image to draw lines on
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Draw the lines on the blank image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 5)
    
    # Combine the line image with the original image
    result = weighted_img(line_img, image, α=0.8, β=1.0, γ=0.)
    
    # Return the original image with lines drawn on it
    return result
    """
    Process a video file to detect lane lines in each frame.
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = process_image(frame)
        
        # Write the frame to the output video
        out.write(processed_frame)
        
        # Display the frame (optional)
        cv2.imshow('Lane Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def test_on_image(filename):
    """
    Test lane detection on a single image and display the result.
    """
    image_path = "input/" + filename

    # Read the image
    image = cv2.imread(image_path)
    
    # Process the image
    result = process_image(image)

    cv2.imwrite('output/' + filename, result)
    
    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection Result')
    plt.axis('off')
    plt.show()
    
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("For images: python lane_detection.py input/[filename]")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Check if the input is an image or video based on the extension
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_on_image(filename)
    else:
        print("Unsupported file format. Please use an image file.")
        sys.exit(1) 