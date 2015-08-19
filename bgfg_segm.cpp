#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

#define M_MOG2 2
#define M_KNN  3

int main(int argc, const char** argv)
{
    CommandLineParser cmd(argc, argv,
        "{ c camera   | false       | use camera }"
        "{ f file     | 768x576.avi | input video file }"
        "{ t type     | mog2        | method's type (knn, mog2) }"
        "{ h help     | false       | print help message }"
        "{ m cpu_mode | false       | press 'm' to switch OpenCL<->CPU}");

    if (cmd.has("help"))
    {
        cout << "Usage : bgfg_segm [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    bool useCamera = cmd.has("camera");
    string file = "/storage/hpc/homes//adarsh/big_buck_bunny.mp4";
//    	string file="/storage/hpc/homes/tejeswini/dataset/video/BOR10_013.mpg";
	string method = cmd.get<string>("type");

    if (method != "mog" && method != "mog2")
    {
        cerr << "Incorrect method" << endl;
        return EXIT_FAILURE;
    }

    int m = method == "mog2" ? M_MOG2 : M_KNN;

    VideoCapture cap;
    if (useCamera)
        cap.open(0);
    else
        cap.open(file);

    if (!cap.isOpened())
    {
        cout << "can not open camera or video file" << endl;
        return EXIT_FAILURE;
    }

    UMat frame, fgmask, fgimg;
    cap >> frame;
    fgimg.create(frame.size(), frame.type());

    Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN();
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();

    switch (m)
    {
    case M_KNN:
        knn->apply(frame, fgmask);
        break;

    case M_MOG2:
        mog2->apply(frame, fgmask);
        break;
    }
    bool running=true;
    double avgfps = 0;
    int ctr = 0;
    int i = 0;
    for (;i<1000;i++)
    {
        if(!running)
            break;
        cap >> frame;
        if (frame.empty())
            break;

        int64 start = getTickCount();

        //update the model
        switch (m)
        {
        case M_KNN:
            knn->apply(frame, fgmask);
            break;

        case M_MOG2:
            mog2->apply(frame, fgmask);
            break;
        }

        double fps = getTickFrequency() / (getTickCount() - start);
        //std::cout << "FPS : " << fps << std::endl;
        //std::cout << fgimg.size() << std::endl;
	ctr++;
	avgfps += fps;
        fgimg.setTo(Scalar::all(0));
        frame.copyTo(fgimg, fgmask);

        imshow("image", frame);
        imshow("foreground mask", fgmask);
        imshow("foreground image", fgimg);

        char key = (char)waitKey(30);

        switch (key)
        {
        case 27:
            running = false;
            break;
        case 'm':
        case 'M':
            ocl::setUseOpenCL(!ocl::useOpenCL());
            cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
            break;
        }
    }
    std::cout << "AVERAGE FPS:" << avgfps/ctr << "\n";;
    return EXIT_SUCCESS;
}
