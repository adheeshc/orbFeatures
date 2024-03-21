#ifndef ORB_FEATURE_MATCHER_CV_H
#define ORB_FEATURE_MATCHER_CV_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cassert>

class ORBFeatureMatcherCV {
private:
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::Ptr<cv::DescriptorMatcher> _matcher;
    void initialize();

public:
    ORBFeatureMatcherCV();
    void extractORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kp, cv::Mat& desc);
    void drawKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& kp, cv::Mat& outImg);
    void matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches);
    void removeOutliers(const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& goodMatches);
    void drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches, cv::Mat& outImg, bool onlyGoodMatches);
};

#endif //ORB_FEATURE_MATCHER_CV_H
