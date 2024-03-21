#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cassert>
#include <cmath>
#include <nmmintrin.h>

#include "ORBFeatureMatcherCV.h"
#include "CustomORBFeatureMatcher.h"

int main() {

    // Load Data
    std::string fileName1 = "../data/1.png";
    std::string fileName2 = "../data/2.png";
    cv::Mat img1 = cv::imread(fileName1);
    cv::Mat img2 = cv::imread(fileName2);
    assert(img1.data != nullptr && img2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<DescType> descriptor1, descriptor2;
    std::vector<cv::DMatch> customMatches;
    
    CustomORBFeatureMatcher customORB;
    customORB.setFASTthreshold(20);

    // Extract ORB Features from both images
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    customORB.extractORB(img1, keypoints1, descriptor1);
    customORB.extractORB(img2, keypoints2, descriptor2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Custom extract ORB cost: " << timeTaken.count() << " seconds" << std::endl;

    // //Use Hamming to draw features
    // cv::Mat outImg1, outImg2;
    // customORB.drawKeypoints(img1, keypoints1, outImg1);
    // customORB.drawKeypoints(img2, keypoints2, outImg2);
    // cv::imshow("ORB Features 1", outImg1);
    // cv::imshow("ORB Features 2", outImg2);
    // cv::waitKey(0);


    //Find Matches
    t1 = std::chrono::steady_clock::now();
    customORB.matchFeatures(descriptor1, descriptor2, customMatches);
    t2 = std::chrono::steady_clock::now();
    timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Custom match ORB cost: " << timeTaken.count() << " seconds" << std::endl;
    std::cout << "Custom matches: " << customMatches.size() << std::endl;

    // plot the matches
    cv::Mat customImgMatches;
    customORB.drawMatches(img1, keypoints1, img2, keypoints2, customMatches, customImgMatches, true);
    cv::imshow("custom Matches", customImgMatches);    

    //Initialize OrbMatcherCV
    ORBFeatureMatcherCV orbMatcherCV;
    cv::Mat descriptors1, descriptors2;
    

    // Extract ORB Features from both images
    t1 = std::chrono::steady_clock::now();
    orbMatcherCV.extractORB(img1, keypoints1, descriptors1);
    orbMatcherCV.extractORB(img2, keypoints2, descriptors2);
    t2 = std::chrono::steady_clock::now();
    timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost: " << timeTaken.count() << " seconds" << std::endl;

    //Use Hamming to draw features
    // cv::Mat outImg1, outImg2;
    // orbMatcherCV.drawKeypoints(img1, keypoints1, outImg1);
    // orbMatcherCV.drawKeypoints(img2, keypoints2, outImg2);
    // cv::imshow("ORB Features 1", outImg1);
    // cv::imshow("ORB Features 2", outImg2);
    // cv::waitKey(0);

    // Match features
    t1 = std::chrono::steady_clock::now();
    std::vector<cv::DMatch> matches;
    orbMatcherCV.matchFeatures(descriptors1, descriptors2, matches);
    t2 = std::chrono::steady_clock::now();
    timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost: " << timeTaken.count() << " seconds" << std::endl;
    std::cout << "matches: " << matches.size() << std::endl;

    //draw results
    cv::Mat imgMatches;
    orbMatcherCV.drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches, true);
    cv::imshow("CV Matches", imgMatches);


    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}