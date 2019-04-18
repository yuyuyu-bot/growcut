#include "growcut.hpp"

static constexpr int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
static constexpr int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
static constexpr float norm = 1.f / (3 * 255 * 255);

namespace segmentation {

growcutImpl::growcutImpl(int rows, int cols, bool visualize)
: rows_(rows), cols_(cols), visualize_(visualize), label_(rows, cols), next_label_(rows, cols), strength_(rows, cols), next_strength_(rows, cols)
{
    strength_.setTo(cv::Scalar(0.0f));
    next_strength_.setTo(cv::Scalar(0.0f));
}

void growcutImpl::initLabels(const cv::Mat& image, const cv::Mat& labels)
{
    double min, max;
    cv::minMaxLoc(labels, &min, &max);
    CV_Assert(static_cast<int>(min) == 0);
    num_labels_ = static_cast<int>(max);

    image.copyTo(labels_vis_);
    labels.copyTo(label_);
    labels.copyTo(next_label_);
    strength_.setTo(cv::Scalar::all(1.0f), labels > 0);
    next_strength_.setTo(cv::Scalar::all(1.0f), labels > 0);
}

inline bool growcutImpl::isValidPosition(int x, int y)
{
    return (x >= 0 && x < cols_ && y >= 0 && y < rows_ && next_label_.ptr<int>(y)[x] != 0);
}

inline void growcutImpl::visLabels()
{
    if (colors_for_vis_.empty()) {
        std::srand((uint32_t)std::time(NULL));
        for (int i = 0; i < num_labels_; i++)
            colors_for_vis_.push_back(cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256));
    }

    labels_vis_.forEach([&](cv::Vec3b& v, const int* p) {
        const int l = label_.ptr<int>(p[0])[p[1]];
        if (l != 0)
            v = colors_for_vis_[l - 1];
    });

    cv::imshow("__growcut_process__", labels_vis_);
    cv::waitKey(1);
}

void growcutImpl::apply(const cv::Mat& image, const cv::Mat& label)
{
    CV_Assert(image.type() == CV_8UC3);
    CV_Assert(label.type() == CV_32S);

    rows_ = image.rows;
    cols_ = image.cols;

    initLabels(image, label);

    int num_updated;
    do {
        num_updated = 0;
        for (int y = 0; y < rows_; ++y) {

            const cv::Vec3b* _image = image.ptr<cv::Vec3b>(y);

            for (int x = 0; x < cols_; ++x) {

                cv::Vec3b center_color = _image[x];

                for (int i = 0; i < 8; i++) {
                    int nx = x + dx[i];
                    int ny = y + dy[i];

                    if (!isValidPosition(nx, ny))
                        continue;

                    cv::Vec3i curr_color = image.ptr<cv::Vec3b>(ny)[nx];
                    float G = 1.f - norm * (
                        (center_color[0] - curr_color[0]) * (center_color[0] - curr_color[0]) +
                        (center_color[1] - curr_color[1]) * (center_color[1] - curr_color[1]) +
                        (center_color[2] - curr_color[2]) * (center_color[2] - curr_color[2]));

                    if (next_strength_.ptr<float>(y)[x] < G * strength_.ptr<float>(ny)[nx]) {
                        if (next_label_.ptr<int>(y)[x] != label_.ptr<int>(ny)[nx])
                            num_updated++;

                        next_strength_.ptr<float>(y)[x] = G * strength_.ptr<float>(ny)[nx];
                        next_label_.ptr<int>(y)[x] = label_.ptr<int>(ny)[nx];
                    }
                }
            }
        }

        memcpy(label_.data, next_label_.data, sizeof(int) * rows_ * cols_);
        memcpy(strength_.data, next_strength_.data, sizeof(float) * rows_ * cols_);

        if (visualize_)
            visLabels();

    } while (num_updated > 0);

    if (visualize_)
        cv::destroyWindow("__growcut_process__");
}

void growcutImpl::getLabels(cv::Mat& label_out) const
{
    label_.copyTo(label_out);
}

void growCut(const cv::Mat& image, cv::Mat& labels, bool visualize_process)
{
    CV_Assert(image.type() == CV_8UC3);
    CV_Assert(labels.type() == CV_32S);

    growcutImpl impl(image.rows, image.cols, visualize_process);
    impl.apply(image, labels);
    impl.getLabels(labels);
}

}