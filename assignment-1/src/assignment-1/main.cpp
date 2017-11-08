/*
  main.cpp
  assignment-1

  The purpose of this file is to take in images of notices, and to find the positions of text within the notices.
 
  Created by Geoffrey Natin on 25/10/2017.
  Copyright © 2017 nating. All rights reserved.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "json.hpp" //https://github.com/nlohmann/json
#include <fstream>

using json = nlohmann::json;
using namespace cv;
using namespace std;

//-------------------------------------------FUNCTIONS TO MAKE USE OF------------------------------------------------------

//This function draws a bounding red rectangle around a contour in an image
void enclose_in_angled_rectangle(cv::Mat img, vector<Point> contour) {
    static cv::RNG rng(12345);
    
    cv::RotatedRect rect = cv::minAreaRect(cv::Mat(contour));
    
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::Point2f rect_points[4];
    rect.points( rect_points );
    
    for (int j = 0; j < 4; j++ ) {
        cv::line(img, rect_points[j], rect_points[(j+1)%4], color, 1, 8);
    }
}

//This function draws a red x-axis aligned rectangle around a contour in an image
void enclose_in_rectangle(cv::Mat img, vector<Point> contour) {
    static cv::RNG rng(12345);
    cv::Rect rect = cv::boundingRect(cv::Mat(contour));
    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (int j = 0; j < 4; j++ ) {
        cv::rectangle(img, rect.tl(), rect.br(), color, 1, 8);
    }
}

//This function draws a red rectangle on an image with its corners in the positions p1 & p2
void draw_rectangle_on_image(cv::Mat img,Point p1, Point p2) {
    static cv::RNG rng(12345);
    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (int j = 0; j < 4; j++ ) {
        cv::rectangle (img, p1, p2, color, 1, 8);
        //cv::line(img, pos[j], pos[(j+1)%4], color, 1, 8);
    }
}

//This function creates images depicting the ground truths for the notices
void create_ground_truth_images(string path_to_images,string path_to_ground_truth_images,json j){
    for (json::iterator it = j["notices"].begin(); it != j["notices"].end(); ++it) {
        string name = it.value()["name"];
        cv::Mat img = imread(path_to_images+name);
        for (json::iterator z = it.value()["boxes"].begin(); z != it.value()["boxes"].end(); ++z) {
            Point p0 = Point(z.value()[0][0],z.value()[0][1]);
            Point p1 = Point(z.value()[1][0],z.value()[1][1]);
            draw_rectangle_on_image(img, p0, p1);
        }
        imwrite(path_to_ground_truth_images+name, img);
    }
}

//This function displays the images from a Mat array in a window (all images must be of the same color space, I think?)
void display_images(int n, cv::Mat images[n]){
    cv::Mat wind = images[0];
    for(int i=1;i<n;i++){
        cv::hconcat(wind,images[i], wind); // horizontal
    }
    std::cout << "Displaying images." << std::endl;
    namedWindow("Window",cv::WINDOW_AUTOSIZE);
    imshow("Window",wind);
}

//This function Grayscales an image
Mat grayscale(Mat img){
    Mat grayscaleMat;
    cvtColor(img, grayscaleMat, CV_RGB2GRAY);
    return grayscaleMat;
}

//This function turns a grayscale image into a binary image
Mat binary(Mat grayscaleImage){
    Mat binaryMat;
    int block_size = 19;
    int offset = 20;
    int output_value = 255;
    adaptiveThreshold( grayscaleImage, binaryMat, output_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, offset );
    return binaryMat;
}

//This function closes a binary image
Mat close(Mat binaryImage){
    int n = 2;
    Mat closedMat;
    Mat n_by_n_element( n, n, CV_8U, Scalar(1) );
    morphologyEx( binaryImage, closedMat, MORPH_CLOSE, n_by_n_element );
    return closedMat;
}

//This function opens a binary image
Mat open(Mat binaryImage){
    int n = 2;
    Mat openedMat;
    Mat n_by_n_element( n, n, CV_8U, Scalar(1) );
    morphologyEx( binaryImage, openedMat, MORPH_OPEN, n_by_n_element );
    return openedMat;
}

//This function takes contours and its hierarchy and returns an array of the indexes of the 'contours with at least n children' within the hierarchy
vector<vector<Point>> getContoursWithNChildren(vector<vector<Point>> contours, vector<Vec4i> hierarchy,int n){
    
    vector<vector<Point>> newContours;
    
    //Find all components with more than 2 children
    if ( !contours.empty() && !hierarchy.empty() ) {
        // loop through the contours
        for ( int i=0; i<contours.size(); i++ ) {
            //If the contour has children, count them
            if(hierarchy[i][2]>0){
                vector<vector<Point>> children;
                //Starting with j as the first child, while there is a next child, move j on and up the count
                int j=hierarchy[i][2];
                while(hierarchy[j][0]>0){
                    children.push_back(contours[j]);
                    j = hierarchy[j][0];
                }
                //If the contour has more than 2 children, add it to newContours
                if (children.size()>n) {
                    newContours.push_back(contours[i]);
                }
            }
        }
    }
    return newContours;
}

//This function takes an image & its contours and draws a rectangle around the contours' points
Mat drawRectangles(Mat img, vector<vector<Point>> contours){
    Mat rectanglesMat = img;
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            enclose_in_rectangle(rectanglesMat,contours[i]);
        }
    }
    return rectanglesMat;
}

//This function takes an image & its contours and draws the contours
Mat drawContours(Mat img, vector<vector<Point>> contours){
    Mat contoursMat = img;
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Scalar colour( (rand()&255), (rand()&255), (rand()&255) );
            drawContours( contoursMat, contours, i, colour);
        }
    }
    return contoursMat;
}



//This function finds the regions of text within an image (under development and doesn't do that at the moment)
void find_text(cv::Mat img){
    
    //Declare variables
    Mat contours_image, croppedImage;
    
    //Convert the image to grayscale
    Mat grayscaleMat = grayscale(img);
    
    //Convert the grayscale image into a binary image
    Mat binaryMat = binary(grayscaleMat);
    
    //Close and Open the image
    Mat closedMat = close(binaryMat);
    Mat openedMat = open(closedMat);
    
    //Find the contours within the image (Connected Components Analysis)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryMat,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    //Find and draw rectangles around the contours with more than 4 children
    vector<vector<Point>> contoursWithNChildren = getContoursWithNChildren(contours, hierarchy, 4);
    
    //Draw the contours of the image
    //Mat signsImg = drawRectangles(img, contoursWithNChildren);
    
    int sp = 30; // 30 or 80 maybe? Tried all tens in between and wasn't much difference
    int sr = 5; // not much difference from 1-10
    Mat img01, img02, img03, img1,img2,img3,img4,img5,img6;
    //cv::pyrMeanShiftFiltering (img, img1, sp, sr, 1, TermCriteria(TermCriteria::MAX_ITER, 5, 1));
    
    Mat poops[] = {img1};
    display_images(1,poops);
    
    //cv::Mat images[] = {grayscaleMat,binaryMat,closedMat,openedMat};
    //display_images(4,images);
    
    waitKey(0);
}

// This the cpp version of the code at (https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv) but it is really bad
void do_the_func(Mat img){
    
    Mat copy = img;
    Mat ret,mask,new_img,dilated;
    
    Mat img2gray = grayscale(img);
    cv::threshold(img2gray, ret, 180, 255, cv::THRESH_BINARY);
    cv::threshold(img2gray, mask, 180, 255, cv::THRESH_BINARY);
    cv::bitwise_and(img2gray, img2gray, copy, mask);
    cv::threshold(copy, ret, 180, 255, cv::THRESH_BINARY);  // for black text , cv.THRESH_BINARY_INV
    cv::threshold(copy, new_img, 180, 255, cv::THRESH_BINARY);  // for black text , cv.THRESH_BINARY_INV
    
    Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS,Point(3,3));  //to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    cv::dilate(new_img, dilated,kernel);  //dilate , more the iteration more the dilation
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(dilated, contours, hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  //get contours
    
    for(int i=0;i<contours.size();i++){
        vector<Point> contour = contours[i];
        //get rectangle bounding contour
        Rect r = cv::boundingRect(contour);
        
        //Don't plot small false positives that aren't text
        if(r.width < 35 and r.height < 35){
            //draw rectangle around contour on original image
            cv::rectangle(img, r.tl(), r.br(), (255, 0, 255), 2);
            
            /*
             //you can crop image and send to OCR  , false detected will return no text :)
             cropped = img_final[y :y +  h , x : x + w]
             
             s = file_name + '/crop_' + str(index) + '.jpg'
             cv2.imwrite(s , cropped)
             index = index + 1
             
             */
            //write original image with added contours to disk
            cv::imshow("captcha_result", img);
            cv::waitKey();
        }
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
        find_text(images[i]);
    }
    
    return 0;
}
