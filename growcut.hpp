#ifndef __GROWCUT_HPP__
#define __GROWCUT_HPP__

#include <opencv2/opencv.hpp>
#include <vector>

namespace segmentation {

class growcutImpl {

public:
    growcutImpl(int rows, int cols, bool visualize = false);
    void apply(const cv::Mat& image, const cv::Mat& labels);
    void getLabels(cv::Mat& label_out) const;

private:
    inline void initLabels(const cv::Mat& image, const cv::Mat& label);
    inline bool isValidPosition(int x, int y);
    inline void visLabels();

    cv::Mat_<int> label_;
    cv::Mat_<int> next_label_;
    cv::Mat_<float> strength_;
    cv::Mat_<float> next_strength_;
    cv::Mat_<cv::Vec3b> labels_vis_;
    std::vector<cv::Vec3b> colors_for_vis_;

    int rows_, cols_;
    int num_labels_;
    bool visualize_;
};

void growCut(const cv::Mat& image, cv::Mat& label, bool visualize_process = false);

}

#endif