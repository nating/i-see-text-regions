/*
  main.cpp
  assignment-1

  The purpose of this file is to take in images of notices, and to find the positions of text within the notices.
 
  Created by Geoffrey Natin on 25/10/2017.
  Copyright © 2017 nating. All rights reserved.
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//This function draws a red rectangle around a contour in an image
void enclose_in_rectangle(cv::Mat img, vector<Point> contour) {
    static cv::RNG rng(12345);
    
    cv::RotatedRect rect = cv::minAreaRect(cv::Mat(contour));
    
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::Point2f rect_points[4];
    rect.points( rect_points );

    for (int j = 0; j < 4; j++ ) {
        cv::line(img, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
    }
}

int main(int argc, const char * argv[]) {
    
    //Load images from their directory
    string path_to_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/notice-images/";
    const int number_of_images = 8;
    Mat images[number_of_images];
    for(int i=0;i<number_of_images;i++){
        string image_name = "Notice"+to_string(i+1)+".jpg";
        images[i] = imread(path_to_images+image_name);
        //imshow(image_name,images[i]);
        //waitKey(0);
    }
    namedWindow("Window",cv::WINDOW_AUTOSIZE);
    imshow("Window",images[0]);
    
    int block_size = 19; // this number must be odd
    int offset = 20; //I made up this number and the one above. You need to mess with them to get them right for certain images apparently
    int output_value = 255; //Apparently this value is usually 255 for binary thresholding (this info is on page 57 of the book)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat contours_image, grayscaleMat, binaryMat, image = images[0];
    cvtColor(image, grayscaleMat, CV_RGB2GRAY);                                         // first convert the image to grayscale
    adaptiveThreshold( grayscaleMat, binaryMat, output_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, offset );
    findContours(binaryMat,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    //contours is an array of lists of boundary points for the boundaries of the various binary regions in the image.
    //hierarchy is an array of indices which allows the contours to be considered in a hierarchical fashion. (https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html#gsc.tab=0)
    //Each contour has an associated entry in the hierarchy and this entry contains four indices:
    //  1.the next contour
    //  2.the previous contour
    //  3.the first child (i.e. enclosed) contour
    //  4.the parent (i.e. enclosing) contour.
    //  Each index is set to a negative number if there is no contour to refer to.
    
    //-----------------
    /*
    //Find the components with more than a certain number of child components
    for (int i=0;i< contours.size();i++)
    {
        int number_of_children = 0;
        current_contour = contours[i];
        while(current_counter){
            
        }
    }
    */
    
    for (int contour=0; (contour < contours.size()); contour++)
    {
        Scalar colour( rand()&0xFF,rand()&0xFF,rand()&0xFF );
        drawContours( images[0], contours, contour, colour, CV_FILLED, 8, hierarchy );
        enclose_in_rectangle(images[0], contours[contour]);
     }
    namedWindow("Window",cv::WINDOW_AUTOSIZE);
    imshow("Window",images[0]);
    waitKey(0);
    std::cout << "Went to here." << std::endl;
    return 0;
}
