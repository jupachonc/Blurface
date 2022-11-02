#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include "omp.h"
#include <cmath>
#include <sys/time.h>

#define R_ARGS 3

// Matrix for effect
int matrixSize1D  = 15;
int fullMatrixSize = 15 * 15;

int numThreads = 1; // Number of threads to use

// Namespace for OpenCV
using namespace cv;

using namespace std;

__global__ void blurImage(short *Matrix, short *rMatrix,
int step, int width, int height, int initX, int initY, int numThreads, int fullMatrixSize, int matrixSize1D){
    
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    int partition = width / numThreads;
    int start_x = threadId * partition;

    int end_x = ((threadId + 1) * partition) - 1;

    

    int max_x = initX + (end_x < width ? end_x : width);
    int max_y = initY + height;

    for (int x = initX + start_x; x <= max_x; x += matrixSize1D)
    {
        for (int y = initY; y <= max_y; y += matrixSize1D)
        {

            
            int new_pixels[3] = {0, 0, 0};
            // Get the positions of all pixels in the group
            for (int i = 0; i < fullMatrixSize; i++)
            {
                int col = x + (i % matrixSize1D);
                int row = y + (int)(i / matrixSize1D);
                
                new_pixels[0] += Matrix[(3 * step * row) + (3 * col) + 0];
                new_pixels[1] += Matrix[(3 * step * row) + (3 * col) + 1];
                new_pixels[2] += Matrix[(3 * step * row) + (3 * col) + 2];

            }

            new_pixels[0] /= fullMatrixSize;
            new_pixels[1] /= fullMatrixSize;
            new_pixels[2] /= fullMatrixSize;

            // Replace the value of all pixels in the group for the previous one calculated
            for (int i = 0; i < fullMatrixSize; i++)
            {
                int col = x + (i % matrixSize1D);
                int row = y + (int)(i / matrixSize1D);
                
                rMatrix[(3 * step * row) + (3 * col) + 0] = (short) new_pixels[0];
                rMatrix[(3 * step * row) + (3 * col) + 1] = (short) new_pixels[1];
                rMatrix[(3 * step * row) + (3 * col) + 2] = (short) new_pixels[2];
            }

        
            
        }
        
    }



};


void detectAndBlur(Mat &img, CascadeClassifier &cascade){
    cudaError_t err = cudaSuccess;
    // Vector to save detected faces coordinates
    vector<Rect> faces;

    // Convert to Gray Scale
    Mat gray;

    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Resize the Grayscale Image
    equalizeHist(gray, gray);

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(gray, faces);

    Mat channels[3];

    split(img, channels);

    Mat B, G, R;

    B = channels[0];
    G = channels[1];
    R = channels[2];

    // Blur detected faces
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        {
            int size = img.total() * img.elemSize();

            //cout << size << endl;

            //cout << img.channels() << endl;

            short *d_Matrix;
            short *h_Matrix;
            short *d_rMatrix;
            short *h_rMatrix;

            h_Matrix = (short *)malloc(size);

            h_rMatrix= (short *)malloc(size);

            h_Matrix = (short *) img.data;

            err = cudaMalloc((void **) &d_Matrix, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device d_B (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }


            err = cudaMemcpy(d_Matrix, h_Matrix, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy Matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMalloc((void **) &d_rMatrix, size);

            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device d_B (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }


            int nBlocks = 80;
            int nThreads = 256;

            blurImage<<<nBlocks, nThreads>>>(d_Matrix, d_rMatrix, img.step, r.width, r.height, r.x, r.y, nBlocks * nThreads, fullMatrixSize, matrixSize1D);

            cudaDeviceSynchronize();

            

            err = cudaMemcpy(h_rMatrix, d_rMatrix, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rMatR from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    img.data = (uchar *)h_rMatrix;

    cudaFree(d_Matrix);
            cudaFree(d_rMatrix);

        
        }
    }
}


int main(int argc, char *argv[]){
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
    numThreads = atoi(*(argv + 3));

    // Verify number of threads
    if (numThreads < 0)
    {
        printf("Invalid threads number \n");
        exit(EXIT_FAILURE);
    }

    // Start time
    gettimeofday(&tval_before, NULL);

    // Force OpenCV use number of threads
    setNumThreads(numThreads);

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

        // Process frame to blur face
        detectAndBlur(frame, cascade);

        // Write proccesed frame in video output
        video.write(frame);

        i++;

        cout << i <<endl;
    }

    cap.release();
    video.release();

    // End time
    gettimeofday(&tval_after, NULL);

    // Calculate time
    timersub(&tval_after, &tval_before, &tval_result);
/*
    printf("\n-----------------------------------------\n");
    printf("Source video: %s\n", loadPath);
    printf("Output video: %s\n", savePath);
    printf("Threads: %d\n", numThreads);
    printf("Execution time: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("\n-----------------------------------------\n");
*/
        return 0;
}

