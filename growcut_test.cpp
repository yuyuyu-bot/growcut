#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

#include "growcut.hpp"

struct mouseParam {
    int x, y, event, flags;
};

static void CallBackFunc(int eventType, int x, int y, int flags, void* userdata)
{
    mouseParam* ptr = static_cast<mouseParam*>(userdata);
    ptr->x = x;
    ptr->y = y;
    ptr->event = eventType;
    ptr->flags = flags;
}

static inline void draw(cv::Mat& image, cv::Mat& labels, const mouseParam& mouseEvent, cv::Vec3b color, int l)
{
    for (int y = std::max(mouseEvent.y - 2, 0); y <= std::min(mouseEvent.y + 2, image.rows - 1); ++y) {
        for (int x = std::max(mouseEvent.x - 2, 0); x <= std::min(mouseEvent.x + 2, image.cols - 1); ++x) {
            image.ptr<cv::Vec3b>(y)[x] = color;
            labels.ptr<int>(y)[x] = l;
        }
    }
}

static void interactiveLabelDrawing(const cv::Mat& image, cv::Mat& labels)
{
    labels.create(image.size(), CV_32S);
    labels.setTo(cv::Scalar::all(0));

    cv::Mat image_show;
    image.copyTo(image_show);

    cv::String window_name = "window";
    cv::imshow(window_name, image_show);

    mouseParam mouseEvent;
    cv::setMouseCallback(window_name, CallBackFunc, &mouseEvent);

    int num = 0;
    std::vector<cv::Vec3b> colors;
    colors.push_back(cv::Vec3b(  0,   0, 255));
    colors.push_back(cv::Vec3b(  0, 255,   0));
    colors.push_back(cv::Vec3b(255,   0,   0));
    colors.push_back(cv::Vec3b(  0, 255, 255));
    colors.push_back(cv::Vec3b(255,   0, 255));
    colors.push_back(cv::Vec3b(255, 255,   0));
    int num_colors = static_cast<int>(colors.size());
    std::printf("label(%d)\r", num + 1);

    bool drawing = false;

    while (true) {
        cv::imshow(window_name, image_show);
        const char c = cv::waitKey(10);

        if (c == 13)
            break;
        else if (c == 'n') {
            num = (num + 1) % num_colors;
            std::printf("label(%d)\r", num + 1);
        }
        else if (c >= '1' && c <= '6') {
            num = static_cast<int>(c - '0') - 1;
            std::printf("label(%d)\r", num + 1);
        }
        else if (c == 27)
            exit(EXIT_SUCCESS);

        if (mouseEvent.event == cv::EVENT_LBUTTONDOWN) {
            while (mouseEvent.event != cv::EVENT_LBUTTONUP && !drawing) {
                cv::imshow(window_name, image_show);
                cv::waitKey(1);
                draw(image_show, labels, mouseEvent, colors[num], num + 1);
                drawing = mouseEvent.event == cv::EVENT_LBUTTONUP;
            }

            drawing = false;
        }
    }

    std::printf("\n");
    cv::destroyWindow(window_name);
}

static cv::Mat1b createContourMask(const cv::Mat& labels)
{
    CV_Assert(labels.type() == CV_32S);

    const int rows = labels.rows;
    const int cols = labels.cols;
    cv::Mat dst(rows, cols, CV_8U, cv::Scalar::all(0));

    dst.forEach<uchar>([&](uchar& v, const int* p) {
        const int x = p[1];
        const int y = p[0];

        const int c = labels.ptr<int>(y)[x];
        const int r = x + 1 < cols ? labels.ptr<int>(y)[x + 1] : -1;
        const int d = y + 1 < rows ? labels.ptr<int>(y + 1)[x] : -1;

        if (r >= 0 && c != r) v = 255u;
        if (d >= 0 && c != d) v = 255u;
    });

    return dst;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: ./growcut_test.exe [image]" << std::endl;
        exit(-1);
    }

    cv::Mat image = cv::imread(argv[1]);
    cv::Mat labels;

    interactiveLabelDrawing(image, labels);
    segmentation::growCut(image, labels);

    cv::Mat labels_contour = createContourMask(labels);
    image.setTo(cv::Scalar(0, 0, 255), labels_contour);

    cv::imshow("GrowCut segment contour", image);
    cv::waitKey(0);

    return 0;
}