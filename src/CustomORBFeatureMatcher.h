#ifndef CUSTOM_ORB_FEATURE_MATCHING_H
#define CUSTOM_ORB_FEATURE_MATCHING_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cassert>
#include <cmath>
#include <nmmintrin.h>

typedef std::vector<uint32_t> DescType;
class CustomORBFeatureMatcher {
private:
    int _threshold = 40;
    void detectKeyPoints(const cv::Mat& img, int threshold, std::vector<cv::KeyPoint>& kp);
    static const int ORB_pattern[256 * 4];

public:
    CustomORBFeatureMatcher() {};
    void setFASTthreshold(int threshold);
    void extractORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kp, std::vector<DescType>& desc);
    void drawKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& kp, cv::Mat& outImg);
    void matchFeatures(const std::vector<DescType>& desc1, const std::vector<DescType>& desc2, std::vector<cv::DMatch>& matches);
    void removeOutliers(const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& goodMatches);
    void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches, cv::Mat& outImg, bool onlyGoodMatches);
};

#endif //CUSTOM_ORB_FEATURE_MATCHING_H


