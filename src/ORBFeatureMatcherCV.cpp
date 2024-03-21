#include "ORBFeatureMatcherCV.h"

ORBFeatureMatcherCV::ORBFeatureMatcherCV() {
    initialize();
}

void ORBFeatureMatcherCV::initialize() {
    _detector = cv::ORB::create();
    _descriptor = cv::ORB::create();
    _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void ORBFeatureMatcherCV::extractORB(const cv::Mat& img, std::vector<cv::KeyPoint>& kp, cv::Mat& desc) {
    assert(!img.empty());
    //Detect OrientedFAST
    _detector->detect(img, kp);
    //compute BRIEF
    _descriptor->compute(img, kp, desc);
}

void ORBFeatureMatcherCV::drawKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& kp, cv::Mat& outImg) {
    cv::drawKeypoints(img, kp, outImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
}

void ORBFeatureMatcherCV::matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches) {
    _matcher->match(desc1, desc2, matches);
}

void ORBFeatureMatcherCV::removeOutliers(const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& goodMatches) {
    auto minMax = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {return m1.distance < m2.distance;});

    double minDist = minMax.first->distance;
    double maxDist = minMax.second->distance;

    // std::cout << "--- Min Dist: " << (double)minDist << "\n";
    // std::cout << "--- Max Dist: " << (double)maxDist << "\n";

    //remove bad matches
    for (const auto& match : matches) {
        if (match.distance <= std::max(2 * minDist, 30.0)) {
            goodMatches.emplace_back(match);
        }
    }
}

void ORBFeatureMatcherCV::drawMatches(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches, cv::Mat& outImg, bool onlyGoodMatches) {
    if (onlyGoodMatches) {
        std::vector<cv::DMatch> goodMatches;
        removeOutliers(matches, goodMatches);
        cv::drawMatches(img1, kp1, img2, kp2, goodMatches, outImg);
    }
    else {
        cv::drawMatches(img1, kp1, img2, kp2, matches, outImg);
    }
}


