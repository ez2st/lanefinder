Lane Detection System
![road](https://github.com/user-attachments/assets/2b09cd89-00dd-41d9-9f16-ce337504d474)
![road2](https://github.com/user-attachments/assets/64f6cf0c-6b9d-4ee1-8523-232dcdf0a9cc)

What is it?

I made a lane detection system that finds and highlights the lane markings on roads in pictures and videos. It uses some computer vision stuff to pick out edges and figure out where the lanes are.

How I Did It

Stuff I Used

C++ to make the main program

Python to quickly test things and visualize results

OpenCV library for handling images

CMake and Make to set everything up

Steps I Followed

1. Getting Started

First, I set up my coding environment. I used CMake and Make because they work well for building projects. For anyone using macOS, I made a quick script (install_opencv_macos.sh) to help install OpenCV easily using Homebrew.

2. Making the Algorithm

Here’s how the algorithm works:

Color Filtering: Picks out the white lane markings.

Noise Reduction: Uses Gaussian blur to smooth the image.

Edge Detection: Finds edges using the Canny edge detection method.

Region of Interest: Looks specifically at the road area.

Line Detection: Finds the lane lines using something called Hough transform.

Visualization: Draws the detected lines onto the original picture.

3. Coding in Two Languages

I wrote this program in both C++ and Python:

The C++ version runs faster and is better for real-time use.

Python was handy for quickly testing things out and visualizing results.

What I Learned

Tech Stuff

OpenCV Tricks: Learned a lot about using OpenCV for image processing, especially finding edges and lines.

Using Multiple Languages: Working in both C++ and Python showed me how performance and speed of development are related.

Build Tools: Figured out how to use CMake and Make to build programs that work on different operating systems.

Computer Vision Stuff

Image Filtering: Discovered how RGB and HSV color spaces help isolate specific things in images.

Edge Detection: Learned how the Canny algorithm identifies edges based on intensity changes.

Hough Transform: Got the hang of using Hough transform to find lines and shapes in images.

Challenges I Faced

Changing Lighting: Handling different lighting was tricky. I solved this by using adaptive thresholding.

Curved Roads: The algorithm works great on straight roads but not so well on curves. I learned this is just how Hough transform is limited.

Speed vs. Accuracy: Had to balance making it fast enough for videos while still being accurate.

Future Ideas

Make the algorithm better at handling curves.

Add warnings when the car starts drifting out of lanes.

Improve speed for smoother real-time video.

Look into deep learning methods to make the detection even better.
