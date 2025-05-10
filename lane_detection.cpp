#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

/**
 * Applies the Canny transform to detect edges
 */
cv::Mat canny(const cv::Mat& img, int low_threshold, int high_threshold) {
    cv::Mat edges;
    cv::Canny(img, edges, low_threshold, high_threshold);
    return edges;
}

/**
 * Applies a Gaussian Noise kernel
 */
cv::Mat gaussian_blur(const cv::Mat& img, int kernel_size) {
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), 0);
    return blurred;
}

/**
 * Applies an image mask.
 * Only keeps the region of the image defined by the polygon
 * formed from vertices. The rest of the image is set to black.
 */
cv::Mat region_of_interest(const cv::Mat& img, const cv::Point* vertices, int vertices_count) {
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    
    // Create a mask with the vertices
    std::vector<std::vector<cv::Point>> roi_poly;
    std::vector<cv::Point> vertices_vec(vertices, vertices + vertices_count);
    roi_poly.push_back(vertices_vec);
    
    // Fill the mask
    if (img.channels() > 1) {
        cv::fillPoly(mask, roi_poly, cv::Scalar(255, 255, 255));
    } else {
        cv::fillPoly(mask, roi_poly, cv::Scalar(255));
    }
    
    // Apply the mask
    cv::Mat masked_image;
    cv::bitwise_and(img, mask, masked_image);
    
    return masked_image;
}

/**
 * Find lines using Hough transform
 */
std::vector<cv::Vec4i> hough_lines(const cv::Mat& img, double rho, double theta, 
                                  int threshold, int min_line_len, int max_line_gap) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(img, lines, rho, theta, threshold, min_line_len, max_line_gap);
    return lines;
}

/**
 * Combine images with weights
 */
cv::Mat weighted_img(const cv::Mat& img, const cv::Mat& initial_img, 
                    double alpha = 0.8, double beta = 1.0, double gamma = 0.0) {
    cv::Mat result;
    cv::addWeighted(initial_img, alpha, img, beta, gamma, result);
    return result;
}

/**
 * Filter the image to only show white-ish areas.
 * This helps to isolate white lane markings.
 */
cv::Mat filter_white(const cv::Mat& img) {
    // Convert to HSV for better color filtering
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    
    // Define range for white color
    cv::Scalar lower_white(0, 0, 180);
    cv::Scalar upper_white(180, 60, 255);
    
    // Create a mask that identifies white pixels
    cv::Mat white_mask;
    cv::inRange(hsv, lower_white, upper_white, white_mask);
    
    // Alternate method: use RGB threshold directly
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    
    // Split channels to apply thresholds individually
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    
    // Apply thresholds to each channel
    cv::Mat r_thresh, g_thresh, b_thresh, rgb_white;
    cv::threshold(channels[0], r_thresh, 220, 255, cv::THRESH_BINARY);
    cv::threshold(channels[1], g_thresh, 220, 255, cv::THRESH_BINARY);
    cv::threshold(channels[2], b_thresh, 220, 255, cv::THRESH_BINARY);
    
    // Combine thresholds
    cv::bitwise_and(r_thresh, g_thresh, rgb_white);
    cv::bitwise_and(rgb_white, b_thresh, rgb_white);
    
    // Combine masks
    cv::Mat combined_mask;
    cv::bitwise_or(white_mask, rgb_white, combined_mask);
    
    // Enhance visibility with morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(combined_mask, combined_mask, kernel);
    
    return combined_mask;
}

/**
 * Process an image to identify and draw lane lines.
 */
cv::Mat process_image(const cv::Mat& image) {
    // Step 1: Filter for white colors (lane markings)
    cv::Mat white_filtered = filter_white(image);
    
    // Step 2: Apply Gaussian blur to reduce noise
    int kernel_size = 5;
    cv::Mat blurred = gaussian_blur(white_filtered, kernel_size);
    
    // Step 3: Apply Canny edge detection
    int low_threshold = 30;
    int high_threshold = 100;
    cv::Mat edges = canny(blurred, low_threshold, high_threshold);
    
    // Step 4: Define region of interest
    int height = image.rows;
    int width = image.cols;
    
    cv::Point vertices[4] = {
        cv::Point(0, height),
        cv::Point(static_cast<int>(width * 0.4), static_cast<int>(height * 0.6)),
        cv::Point(static_cast<int>(width * 0.6), static_cast<int>(height * 0.6)),
        cv::Point(width, height)
    };
    
    cv::Mat masked_edges = region_of_interest(edges, vertices, 4);
    
    // Step 5: Apply Hough transform to detect lines
    double rho = 2;
    double theta = CV_PI / 180;
    int threshold = 20;
    int min_line_length = 30;
    int max_line_gap = 20;
    std::vector<cv::Vec4i> lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap);
    
    // Create a blank color image to draw lines on
    cv::Mat line_img = cv::Mat::zeros(image.size(), CV_8UC3);
    
    // Draw the lines on the blank image
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(line_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 5);
    }
    
    // Combine the line image with the original image
    cv::Mat result = weighted_img(line_img, image);
    
    return result;
}

/**
 * Test lane detection on a single image and display the result.
 */
void test_on_image(const std::string& image_path) {
    // Read the image
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        std::cerr << "Error: Could not read image: " << image_path << std::endl;
        return;
    }
    
    // Process the image
    cv::Mat result = process_image(image);
    
    // Display the result
    cv::namedWindow("Lane Detection Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Lane Detection Result", result);
    
    // Save result
    std::string output_path = image_path.substr(0, image_path.find_last_of('.')) + "_result.jpg";
    cv::imwrite(output_path, result);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "For images: " << argv[0] << " path/to/image" << std::endl;
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string ext = input_path.substr(input_path.find_last_of('.') + 1);
    
    // Convert extension to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "jpg" || ext == "jpeg" || ext == "png") {
        test_on_image(input_path);
    } else {
        std::cout << "Unsupported file format. Please use an image file." << std::endl;
        return 1;
    }
    
    return 0;
} 