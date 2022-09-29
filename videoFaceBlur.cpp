#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <sys/time.h>

#define MATRIX_SIZE_1D 20
#define FULL_MATRIX_SIZE MATRIX_SIZE_1D *MATRIX_SIZE_1D
#define NUM_THREADS 1 // Number of threads to use
#define R_ARGS 2

// Namespace for OpenCV
using namespace cv;

using namespace std;

// Function for Face Detection
void detectAndBlur(Mat &img, CascadeClassifier &cascade);

// Function for blur image
void blurImage(Mat frame, Rect face);

int main(int argc, char *argv[])
{
    // Time values
    struct timeval tval_before, tval_after, tval_result;

    // Initialize strings for paths
    char *loadPath, *savePath;

    // Verify amount of arguments
    if ((argc - 1) < R_ARGS)
    {
        printf("%d arguments are required for execution\n", R_ARGS);
        exit(1);
    }

    // Read args
    loadPath = *(argv + 1);
    savePath = *(argv + 2);

    // Start time
    gettimeofday(&tval_before, NULL);

    // Force OpenCV use number of threads
    setNumThreads(NUM_THREADS);

    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade;

    // Load casacade definer
    cascade.load("./HAARFiles/haarcascade_frontalface_default.xml");

    VideoCapture cap(loadPath);

    // Data source video
    int fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Create video writer
    VideoWriter video(savePath, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    int frame_count;
    frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    // cap is the object of class video capture that tries to capture Bumpy.mp4
    if (!cap.isOpened()) // isOpened() returns true if capturing has been initialized.
    {
        cout << "Cannot open the video file. \n";
        exit(1);
    }

    // double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    //  The function get is used to derive a property from the element.
    //  Example:
    //  CV_CAP_PROP_POS_MSEC : Current Video capture timestamp.
    //  CV_CAP_PROP_POS_FRAMES : Index of the next frame.

    int i = 0;
    while (i < frame_count)
    {
        cap.set(CAP_PROP_POS_FRAMES, i);
        Mat frame;
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout << "\n Cannot read the video file. \n";
            exit(1);
        }

        // Process frame to blur face
        detectAndBlur(frame, cascade);

        // Write proccesed frame in video output
        video.write(frame);

        i++;
    }

    cap.release();
    video.release();

    // End time
    gettimeofday(&tval_after, NULL);

    // Calculate time
    timersub(&tval_after, &tval_before, &tval_result);

    printf("Source video: %s\n", loadPath);
    printf("Output video: %s\n", savePath);
    printf("Execution time: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    return 0;
}

void detectAndBlur(Mat &img, CascadeClassifier &cascade)
{
    // Vector to save detected faces coordinates
    vector<Rect> faces;

    // Convert to Gray Scale
    Mat gray;

    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Resize the Grayscale Image
    equalizeHist(gray, gray);

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(gray, faces);

    // Blur detected faces
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];

        blurImage(img, r);
    }
}

void blurImage(Mat frame, Rect face)
{
    vector<tuple<Point2d, Vec3b>> distorced_face;

    int max_x = face.x + face.width;
    int max_y = face.y + face.height;
    for (int x = face.x; x <= max_x; x += MATRIX_SIZE_1D)
    {
        for (int y = face.y; y <= max_y; y += MATRIX_SIZE_1D)
        {
            // Allocate memmory for store the positions of the pixels in the group
            Point2d *pixels_position = (Point2d *)malloc(sizeof(Point2d) * FULL_MATRIX_SIZE);
            if (pixels_position == NULL)
            {
                perror("Error allocating memory for pixels positions");
                exit(EXIT_FAILURE);
            }

            // Get the positions of all pixels in the group
            for (int i = 0; i < FULL_MATRIX_SIZE; i++)
            {
                *(pixels_position + i) = Point(x + (i % MATRIX_SIZE_1D), y + (int)(i / MATRIX_SIZE_1D));
            }

            // Calculate the average value of the grouped pixels
            int new_pixels[3] = {0, 0, 0};
            for (int i = 0; i < FULL_MATRIX_SIZE; i++)
            {
                Point2d *pixel_position = pixels_position + i;
                Vec3b pixel = frame.at<Vec3b>(pixel_position->y, pixel_position->x);

                new_pixels[0] += pixel[0];
                new_pixels[1] += pixel[1];
                new_pixels[2] += pixel[2];
            }
            new_pixels[0] /= FULL_MATRIX_SIZE;
            new_pixels[1] /= FULL_MATRIX_SIZE;
            new_pixels[2] /= FULL_MATRIX_SIZE;

            // Replace the value of all pixels in the group for the previous one calculated
            for (int i = 0; i < FULL_MATRIX_SIZE; i++)
            {
                Point2d *pixel_position = pixels_position + i;
                Vec3b &pixel = frame.at<Vec3b>(pixel_position->y, pixel_position->x);

                pixel.val[0] = new_pixels[0];
                pixel.val[1] = new_pixels[1];
                pixel.val[2] = new_pixels[2];
            }

            // Free memory
            free(pixels_position);
        }
    }
}