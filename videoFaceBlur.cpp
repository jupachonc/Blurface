#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>

#define R_ARGS 2

// Matrix for effect
int matrixSize1D = 15;
int fullMatrixSize = 15 * 15;

int numBlocks = 1;
int numThreads = 1; // Number of threads to use

// Namespace for OpenCV
using namespace cv;

using namespace std;

void blurImage(uchar *Matrix, uchar *rMatrix, int step, int width, int height, int size)
{
    // Variables for MPI
    int n, processId, numProcs;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    // Brodcast image data from root node
    MPI_Bcast(Matrix, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Calculate iterations for node
    int start_y = ((height) / numProcs) * processId;
    int end_y = (processId < (height) % numProcs) ? start_y + ((height) / numProcs) : start_y + ((height) / numProcs) - 1;
    int max_x = width;

    // Calculate matriz size for allocation
    int start_Mat = 3 * step * start_y;
    int end_Mat = (3 * step * end_y) + (3 * width);
    int matSize = end_Mat - start_Mat;

    // Allocate memory and wait all nodes
    uchar *rpMatrix = new uchar[matSize];
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    for (int y = start_y; y <= end_y; y += matrixSize1D)
    {

        for (int x = 0; x < max_x; x += matrixSize1D)
        {

            // Create new pixels
            int new_pixels[3] = {0, 0, 0};

            // Read matrix values
            for (int i = 0; i < fullMatrixSize; i++)
            {
                int col = x + (i % matrixSize1D);
                int row = y + (int)(i / matrixSize1D);

                // Get pixels info
                if (((3 * step * row) + (3 * col) + 2) <= end_Mat)
                {

                    new_pixels[0] += Matrix[(3 * step * row) + (3 * col) + 0];
                    new_pixels[1] += Matrix[(3 * step * row) + (3 * col) + 1];
                    new_pixels[2] += Matrix[(3 * step * row) + (3 * col) + 2];
                }
            }

            // Calcule final pixel values

            new_pixels[0] /= fullMatrixSize;
            new_pixels[1] /= fullMatrixSize;
            new_pixels[2] /= fullMatrixSize;

            // Replace the value of all pixels in the group for the previous one calculated
            for (int i = 0; i < fullMatrixSize; i++)
            {
                int col = x + (i % matrixSize1D);
                int row = y - start_y + (int)(i / matrixSize1D);

                // Asign new pixels to img
                if (((3 * step * row) + (3 * col) + 2) <= matSize)
                {
                    rpMatrix[(3 * step * row) + (3 * col) + 0] = (uchar)new_pixels[0];
                    rpMatrix[(3 * step * row) + (3 * col) + 1] = (uchar)new_pixels[1];
                    rpMatrix[(3 * step * row) + (3 * col) + 2] = (uchar)new_pixels[2];
                }
            }
        }
    }

    // Wait and collect data from all nodes
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    MPI_Gather(rpMatrix, matSize, MPI_UNSIGNED_CHAR, rMatrix, matSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    // Free memory of local matrix
    free(rpMatrix);
};

void detectAndBlur(Mat &img, Mat &imgBlur, CascadeClassifier &cascade)
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
        {

            // Copy data from blurred img to original
            Mat face = imgBlur(r);
            face.copyTo(img(r));
        }
    }
}

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
    // setNumThreads(numThreads);

    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade;

    // Load casacade definer
    cascade.load("./HAARFiles/haarcascade_frontalface_default.xml");

    VideoCapture cap(loadPath);

    // Data source video
    int fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Determine matrix size
    int matrixSize1D = ceil(height * 0.015);
    int fullMatrixSize = matrixSize1D * matrixSize1D;

    // Create video writer
    VideoWriter video(savePath, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    int frame_count;
    frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    // cap is the object of class video capture that tries to capture Bumpy.mp4
    if (!cap.isOpened()) // isOpened() returns true if capturing has been initialized.
    {
        perror("Cannot open the video file. \n");
        exit(EXIT_FAILURE);
    }

    // double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    //  The function get is used to derive a property from the element.
    //  Example:
    //  CV_CAP_PROP_POS_MSEC : Current Video capture timestamp.
    //  CV_CAP_PROP_POS_FRAMES : Index of the next frame.

    // Init MPI Process
    MPI_Init(NULL, NULL);

    // Get MPI Variables
    int processId, numProcs;

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Iterate video
    int i = 0;
    while (i < frame_count)
    {
        cap.set(CAP_PROP_POS_FRAMES, i);
        Mat frame;
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            perror("\n Cannot read the video file. \n");
            exit(EXIT_FAILURE);
        }

        // clone img to blur
        Mat frameBlurred = frame.clone();

        // Get img size
        int size = frame.total() * frame.elemSize();

        // Define data for apply filter
        uchar *Matrix;
        uchar *rMatrix;

        Matrix = frameBlurred.data;

        rMatrix = (uchar *)malloc(size);

        blurImage(Matrix, rMatrix, (frame.step / frame.elemSize()), frame.cols, frame.rows, size);

        // Replace filter data in fra,e
        frameBlurred.data = rMatrix;

        // Applicate new img to faces
        if (processId == 0)
        {
            // Process frame to blur face
            detectAndBlur(frame, frameBlurred, cascade);
            // Write proccesed frame in video output
            video.write(frame);
        }

        i++;
    }

    // Release video 
    cap.release();
    video.release();

    // Calculate time 
    if (processId == 0)
    {

        // End time
        gettimeofday(&tval_after, NULL);

        // Calculate time
        timersub(&tval_after, &tval_before, &tval_result);

        printf("\n-----------------------------------------\n");
        printf("Source video: %s\n", loadPath);
        printf("Output video: %s\n", savePath);
        printf("Process: %d\n", numProcs);
        printf("Execution time: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        printf("\n-----------------------------------------\n");
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
