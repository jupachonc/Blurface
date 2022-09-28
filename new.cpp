#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
// highgui - an interface to video and image capturing.
#include <opencv2/imgproc/imgproc.hpp> // For dealing with images
#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
// The header files for performing input and output.

using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;
// For input output operations.


// Function for Face Detection
void detectAndDraw(Mat &img, CascadeClassifier &cascade, double scale);
string cascadeName, nestedCascadeName;


void blurImage(Mat matImg, Rect face)
{
    std::vector<Point2d, Vec3b> distorced_face;

    int max_x = face.x + face.width;
    int max_y = face.y + face.height;
    for(int x = face.x; x < max_x; x++){
        for (int y=face.y; y < max_y; y++){
            int new_b = 0;
            int new_g = 0;
            int new_r = 0;
            for(int xf= x - 10; xf <= x+10; x++){
                for(int yf= y - 10; yf <= y+10;y++){
                    cv::Vec3b pixel = matImg.at<Vec3b>(xf, yf);
                    new_b += pixel.val[0];
                    new_g += pixel.val[1];
                    new_r += pixel.val[2];

                }
            }
            

            Vec3b new_pixel = Vec3b(new_b/ 10 * 10, new_g/ 10 * 10, new_r/ 10 * 10);
            distorced_face.push_back(make_tuple(Point2d(x, y), new_pixel));
        }
    }


    for(<Point2d, Vec3b> nPixel : distorced_face){
        matImg.at<Vec3b>get<0>(nPixel) = get<1>(nPixel);
    }
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

        blurImage(img, r);
    }

    // Show Processed Image with detected faces
}
