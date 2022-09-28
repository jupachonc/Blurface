#include "opencv2/highgui/highgui.hpp"
// highgui - an interface to video and image capturing.
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
// The header files for performing input and output.

using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;
// For input output operations.

// Function for Face Detection
void detectAndDraw(Mat &img, CascadeClassifier &cascade, double scale);
string cascadeName, nestedCascadeName;

int accessPixel(unsigned char *arr, int col, int row, int k, int width, int height)
{
    int sum = 0;
    int sumKernel = 0;

    for (int j = -1; j <= 1; j++)
    {
        for (int i = -1; i <= 1; i++)
        {
            if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
            {
                int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
                sum += color;
            }
        }
    }

    return sum / (15 * 15);
}

int kernel[3][3] = {1, 2, 1,
                    2, 4, 2,
                    1, 2, 1};

void guassian_blur2D(unsigned char *arr, unsigned char *result, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            for (int k = 0; k < 3; k++)
            {
                result[15 * row * width + 3 * col + k] = accessPixel(arr, col, row, k, width, height);
            }
        }
    }
}

void blurImage(Mat matImg)
{
    uint8_t *pixelPtr = (uint8_t *)matImg.data;
    int cn = matImg.channels();
    Scalar_<uint8_t> bgrPixel;
    Scalar_<uint8_t> bgrPixelResult;

    /*Crear las tres matrices para cada canal de color*/
    unsigned char *matR, *matG, *matB;
    /*Crear las tres matrices resultanres*/
    unsigned char *rMatR, *rMatG, *rMatB;

    for (int i = 0; i < matImg.rows; i++)
    {
        for (int j = 0; j < matImg.cols; j++)
        {
            bgrPixel.val[0] = pixelPtr[i * matImg.cols * cn + j * cn + 0]; // B
            bgrPixel.val[1] = pixelPtr[i * matImg.cols * cn + j * cn + 1]; // G
            bgrPixel.val[2] = pixelPtr[i * matImg.cols * cn + j * cn + 2]; // R

            *(matB + ((i * matImg.cols) + j)) = bgrPixel.val[0];
            *(matG + ((i * matImg.cols) + j)) = bgrPixel.val[1];
            *(matR + ((i * matImg.cols) + j)) = bgrPixel.val[2];


            // do something with BGR values...
        }
    }

    guassian_blur2D(matR, rMatR, matImg.cols, matImg.rows);
    guassian_blur2D(matG, rMatG, matImg.cols, matImg.rows);
    guassian_blur2D(matB, rMatB, matImg.cols, matImg.rows);

    cout << bgrPixel.val[0] << endl;
}

int main()
{
    // setNumThreads(1);

    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    cascade.load("./haarcascade_frontalface_default.xml");

    VideoCapture cap("video(1).mp4");

    double fps = cap.get(CAP_PROP_FPS);

    VideoWriter video("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(1280, 720));
    int frame_count;
    frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    cout << "Frame count: " << frame_count << endl;
    // cap is the object of class video capture that tries to capture Bumpy.mp4
    if (!cap.isOpened()) // isOpened() returns true if capturing has been initialized.
    {
        cout << "Cannot open the video file. \n";
        return -1;
    }

    // double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    //  The function get is used to derive a property from the element.
    //  Example:
    //  CV_CAP_PROP_POS_MSEC : Current Video capture timestamp.
    //  CV_CAP_PROP_POS_FRAMES : Index of the next frame.

    int i;
    i = 0;
    while (i < frame_count)
    {
        cap.set(CAP_PROP_POS_FRAMES, i);
        Mat frame;
        // Mat object is a basic image container. frame is an object of Mat.
        if (!cap.read(frame)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout << "\n Cannot read the video file. \n";
            break;
        }

        detectAndDraw(frame, cascade, scale);
        // blur(frame,frame,Size(10,10)); // To blur the image.
        video.write(frame);

        i++;
    }

    cap.release();
    video.release();

    return 0;
}

void detectAndDraw(Mat &img, CascadeClassifier &cascade,
                   double scale)
{
    vector<Rect> faces;
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale
    double fx = 1 / scale;

    // Resize the Grayscale Image
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.1,
                             4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Draw circles around the faces

    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];

        blurImage(img(r));
    }

    // Show Processed Image with detected faces
}
