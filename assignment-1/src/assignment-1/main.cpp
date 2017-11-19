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

//-------------------------------------------- VISION TECHNIQUES -------------------------------------------------

//This function Grayscales an image
Mat grayscale(Mat img){
    Mat grayscaleMat;
    cvtColor(img, grayscaleMat, CV_RGB2GRAY);
    return grayscaleMat;
}

//This function turns a grayscale image into a binary image using apaptiveThresholding
Mat adaptiveBinary(Mat grayscaleImage,int block_size,int offset, int output_value){
    Mat binaryMat;
    adaptiveThreshold( grayscaleImage, binaryMat, output_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, offset );
    return binaryMat;
}

//This function turns a grayscale image into a binary image with a threshold
Mat binary(Mat grayscaleImage,int block_size,int offset, int output_value){
    Mat binaryMat;
    int threshold = 128; //Shouldn't matter as OTSU is used
    cv::threshold( grayscaleImage, binaryMat, threshold, 255, THRESH_BINARY | THRESH_OTSU );
    return binaryMat;
}

//This function closes a binary image using an nxn element
Mat close(Mat binaryImage,int n){
    Mat closedMat;
    Mat n_by_n_element( n, n, CV_8U, Scalar(1) );
    morphologyEx( binaryImage, closedMat, MORPH_CLOSE, n_by_n_element );
    return closedMat;
}

//This function opens a binary image with an using an nxn element
Mat open(Mat binaryImage,int n){
    Mat openedMat;
    Mat n_by_n_element( n, n, CV_8U, Scalar(1) );
    morphologyEx( binaryImage, openedMat, MORPH_OPEN, n_by_n_element );
    return openedMat;
}

//This function takes an image and the accepted color difference for pixels to be in the same region, returning a flood filled version of the image
Mat floodFill( Mat img, int color_difference){
    CV_Assert( !img.empty() );
    Mat flooded = img.clone();
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    for(int y=0;y<flooded.rows;y++){
        for(int x=0;x<flooded.cols;x++){
            if( mask.at<uchar>(y+1, x+1) == 0 ){
                //Scalar newVal( rng(256), rng(256), rng(256) );
                Point point(x,y);
                Point3_<uchar>* p = img.ptr<Point3_<uchar>>(y,x);
                Scalar pointColour(p->x,p->y,p->z);
                floodFill( flooded, mask, Point(x,y), pointColour, 0, Scalar::all(color_difference), Scalar::all(color_difference),4);
            }
        }
    }
    return flooded;
}

//This function takes an image and mean-shift parameters and returns a version of the image that has had mean shift segmentation performed on it
Mat meanShiftSegmentation( Mat img, int spatial_radius, int color_radius, int maximum_pyramid_level ){
    Mat res = img.clone();
    pyrMeanShiftFiltering(img,res,spatial_radius,color_radius,maximum_pyramid_level);
    return res;
}

//-------------------------------------------FUNCTIONS TO MAKE USE OF------------------------------------------------------

//This class represents the bounding rectangle of a segment from a mean-shift-segmented image. It has the bounding rectangle and the color of the segment.
class segmentRectangle{
    public:
        Scalar color;
        Rect rect;
        //Overload == operator for segmentRectangle
        friend bool operator==(const segmentRectangle &lhs, const segmentRectangle &rhs){
            return (lhs.color==rhs.color && lhs.rect==rhs.rect);
        }
};

//This function returns true if the two rectangles intersect
bool intersect(Rect r1, Rect r2){
    return ((r1 & r2).area() > 0);
}

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

vector<vector<Rect>> read_ground_truths(string path_to_ground_truths_json){
    vector<vector<Rect>> ground_truths;
    std::ifstream i(path_to_ground_truths_json);
    json j;
    i >> j;
    for (json::iterator it = j["notices"].begin(); it != j["notices"].end(); ++it) {
        vector<Rect> gs;
        string name = it.value()["name"];
        for (json::iterator z = it.value()["boxes"].begin(); z != it.value()["boxes"].end(); ++z) {
            Point p0 = Point(z.value()[0][0],z.value()[0][1]);
            Point p1 = Point(z.value()[1][0],z.value()[1][1]);
            gs.push_back(Rect(p0,p1));
        }
        ground_truths.push_back(gs);
    }
    return ground_truths;
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
    Mat rectanglesMat = img.clone();
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            enclose_in_angled_rectangle(rectanglesMat, contours[i]);
        }
    }
    return rectanglesMat;
}

//This function gets the average pixel value of a component in an image


//This function takes an image & its contours and outlines the contours with random colors
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

//This function displays the images from a Mat array in a window (all images must be of the same color space, I think?)
void display_images(string window_name, int n, cv::Mat images[n]){
    cv::Mat wind = images[0];
    for(int i=1;i<n;i++){
        cv::hconcat(wind,images[i], wind); // horizontal
    }
    namedWindow(window_name,cv::WINDOW_AUTOSIZE);
    imshow(window_name,wind);
}

//This function takes an image & its contours and fills each contour with its average pixel value
Mat fillContours(Mat img, vector<vector<Point>> contours){
    Mat contoursMat = img.clone();
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            //Create mask to find average pixel value in original image
            Mat labels = cv::Mat::zeros(img.size(), CV_8UC1);
            drawContours(labels, contours, i, Scalar(255),CV_FILLED);
            Scalar average_color = mean(img, labels);
            //Fill the contour with the average value of its pixels
            drawContours( contoursMat, contours, i, average_color,CV_FILLED);
        }
    }
    return contoursMat;
}

//This function takes an image and the accepted color difference for pixels to be in the same region, then returns an image with each segment enclosed with a bounding rectangle of its color.
Mat getSegmentsRectanglesImage( Mat img, int color_difference){
    Mat newImg = img.clone();
    CV_Assert( !img.empty() );
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if( mask.at<uchar>(y+1, x+1) == 0 ){
                Rect boundingRectOfSegment;
                Point point(x,y);
                Point3_<uchar>* p = img.ptr<Point3_<uchar>>(y,x);
                Scalar pointColour(p->x,p->y,p->z);
                floodFill( img, mask, Point(x,y), pointColour, &boundingRectOfSegment, Scalar::all(color_difference), Scalar::all(color_difference),4);
                rectangle(newImg, boundingRectOfSegment, pointColour);
            }
        }
    }
    return newImg;
}

//This function takes an image and the accepted color difference for pixels to be in the same region, then returns an array of segmentRectangles in the image each of which is the bounding rectangle of a segment in the image, with the color of that segment.
vector<segmentRectangle> getSegmentsRectangles( Mat img, int color_difference, int min_rect_width, int min_rect_height){
    vector<segmentRectangle> srs;
    Mat discardableImg = img.clone();
    CV_Assert( !img.empty() );
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if( mask.at<uchar>(y+1, x+1) == 0 ){
                Rect boundingRectOfSegment;
                Point point(x,y);
                Point3_<uchar>* p = img.ptr<Point3_<uchar>>(y,x);
                Scalar pointColour(p->x,p->y,p->z);
                floodFill(discardableImg, mask, Point(x,y), pointColour, &boundingRectOfSegment, Scalar::all(color_difference), Scalar::all(color_difference),4);
                if(boundingRectOfSegment.width >= min_rect_width && boundingRectOfSegment.height >= min_rect_height){
                    segmentRectangle segmentRect;
                    segmentRect.color = pointColour;
                    segmentRect.rect = boundingRectOfSegment;
                    srs.push_back(segmentRect);
                }
            }
        }
    }
    return srs;
}

//This function takes an image and the accepted color difference for pixels to be in the same region, then returns an array of segmentRectangles in the image each of which is the bounding rectangle of a segment in the image, with the color of that segment.
vector<segmentRectangle> getSegmentsRectanglesWithContours( Mat img, vector<vector<Point>> contours, vector<Vec4i> hierarchy, int color_difference, int min_rect_width, int min_rect_height){
    
    int max_children_for_letter_contour = 4;
    vector<segmentRectangle> srs;
    CV_Assert( !img.empty() );
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            //Create mask to find average pixel value of the contour in the original image
            Mat labels = cv::Mat::zeros(img.size(), CV_8UC1);
            drawContours(labels, contours, i, Scalar(255),CV_FILLED);
            //A Problem here is that the mask allows everything inside the outline of the contour *including the contour's children* because we have made all pixels inside it white.
            //It should not allow through the contour's children. To fix this problem, we will fill all the children of the contour in black again.
            //The average color of the segment only really matters if it could be a character, so we will only do this for contours with less than 4 children as it is computationally expensive
            //Navigate the children in the hierarchy and fill them black in the mask.
            if (!hierarchy.empty()) {
                //If the contour has children, fill each one black on the mask
                if(hierarchy[i][2]>0){
                    //Starting with j as the first child, while there is a next child, move j on and fill the mask
                    int j=hierarchy[i][2];
                    int contour_children = 1;
                    while(hierarchy[j][0]>0 && contour_children<=max_children_for_letter_contour){
                        contour_children++;
                        j = hierarchy[j][0];
                    }
                    if(contour_children<=max_children_for_letter_contour){
                        int j=hierarchy[i][2];
                        drawContours(labels, contours, j, Scalar(0),CV_FILLED);
                        while(hierarchy[j][0]>0){
                            drawContours(labels, contours, j, Scalar(0),CV_FILLED);
                            j = hierarchy[j][0];
                        }
                    }
                }
            }
            Scalar average_color = mean(img, labels);
            
            //Get the bounding rectangle of the contour
            Rect boundingRectOfContour = boundingRect(contours[i]);
            if(boundingRectOfContour.width >= min_rect_width && boundingRectOfContour.height >= min_rect_height){
                segmentRectangle segmentRect;
                segmentRect.color = average_color;
                segmentRect.rect = boundingRectOfContour;
                srs.push_back(segmentRect);
            }
        }
    }
    return srs;
}

//
vector<Rect> encloseIntersectingRects(vector<Rect> textRegions){
    
    vector<Rect> intersectFinishedTextRegions;
    
    //Create graphs to represent the text regions
    int textRegionGraph[textRegions.size()];
    for(int i=0;i<textRegions.size();i++){
        textRegionGraph[i] = i;
    }
    
    //Connect graphs of intersecting text regions
    for(int i=0;i<textRegions.size();i++){
        int i_region = textRegionGraph[i];
        for(int j=0;j<textRegions.size();j++){
            int j_region = textRegionGraph[j];
            if(intersect(textRegions[i], textRegions[j])){
                //Add every rectangle k, from j's region, to rectangle i's region
                for(int k=0;k<textRegions.size();k++){
                    if(textRegionGraph[k]==j_region){ textRegionGraph[k] = i_region; }
                }
            }
        }
    }
    
    //For all text region that are connected, get the enclosing rectangle of the text region
    for (int i=0;i<textRegions.size();i++) {
        int i_region = textRegionGraph[i];
        if(i_region<0){ continue; }
        int tlx = textRegions[i].tl().x; int tly = textRegions[i].tl().y; int brx = textRegions[i].br().x; int bry = textRegions[i].br().y;
        //Get the maximum tl and br of the text region
        for(int j=0;j<textRegions.size();j++){
            if(textRegionGraph[j]==textRegionGraph[i]){
                tlx = min(tlx, textRegions[j].tl().x);
                tly = min(tly, textRegions[j].tl().y);
                brx = max(brx, textRegions[j].br().x);
                bry = max(bry, textRegions[j].br().y);
            }
        }
        cv::Rect boundingRect(Point(tlx,tly),Point(brx,bry));
        intersectFinishedTextRegions.push_back(boundingRect);
        for(int j=0;j<textRegions.size();j++){
            if(textRegionGraph[j]==i_region){ textRegionGraph[j] = -1; }
        }
    }
    return intersectFinishedTextRegions;
}

//------------------------------------------------ FINDING TEXT SPECIFICALLY --------------------------------------------------------

//This function returns true if the second rectangle is enclosed within the first
bool isEnclosing(Rect r1, Rect r2){
    return ( r2.tl().x > r1.tl().x && r2.tl().y > r1.tl().y && r2.br().x < r1.br().x && r2.br().y < r2.br().y );
}

//This function checks that two rectangles are close by to eachother
bool areCloseBy(Rect r1, Rect r2,float height_increase_for_overlap,float width_increase_for_overlap){
    Point newTL(r1.tl().x-(width_increase_for_overlap*(r1.width/2)),r1.tl().y-(height_increase_for_overlap*(r1.height/2)));
    Point newBR(r1.br().x+(width_increase_for_overlap*(r1.width/2)),r1.br().y+(height_increase_for_overlap*(r1.height/2)));
    Rect r3(newTL,newBR);
    return intersect(r3, r2) && !intersect(r1,r2);
}

//This function takes two RGB colors and returns whether they are within the allowed euclidean color difference within the RGB space
bool areTheSameColor(Scalar c1, Scalar c2,float allowed_color_difference){
    cv::Vec4d d = c1-c2;
    double distance = cv::norm(d);
    return distance <= allowed_color_difference;
}

//This function takes two rectangles and checks that they are rougly the same size
bool areRoughlyTheSameSize(Rect r1, Rect r2,float allowed_height_ratio, float allowed_width_ratio){
    int h1 = r1.height;
    int h2 = r2.height;
    int w1 = r1.width;
    int w2 = r2.width;
    return(
               (    //Are within allowed height ratio of eachother
                     (h1<=allowed_height_ratio*h2 && h1>=h2)
                   ||(h2<=allowed_height_ratio*h1 && h2>=h1)
                )
               &&
               (    //Are within allowed width ratio of eachother
                     (w1<=allowed_width_ratio*w2 && w1>=w2)
                   ||(w2<=allowed_width_ratio*w1 && w2>=w1)
                )
           );
}

//This function takes two rectangles and determines whether they could be considered two letters of a word
//  (are nearby, roughly same size, same color)
bool couldBeTwoLetters(Rect r1, Rect r2,Scalar r1_color,Scalar r2_color){
    
    //highscore: 0.916105
    float allowed_height_ratio = 1.82; // 1.8 splits up the s in 'This' and the g in 'grass' on last notice. 1.85 puts together the second and third sign on notice 4 (the blue 3)
    float allowed_width_ratio = 2.5;
    float height_increase_for_overlap = 0;//1.5;
    float width_increase_for_overlap = 9;//3.5;
    float allowed_color_difference = 30;
    
    return areRoughlyTheSameSize(r1,r2,allowed_height_ratio,allowed_width_ratio)
                && areCloseBy(r1, r2,height_increase_for_overlap,width_increase_for_overlap)
                    && areTheSameColor(r1_color, r2_color,allowed_color_difference);
}

//This function takes
vector<segmentRectangle> getLettersInMyRegion(vector<segmentRectangle> segmentRects,int current_index,bool* inATextRegion, Mat img,Scalar newVal){
    
    vector<int> lettersInMyRegionIndexes;
    vector<segmentRectangle> lettersInMyRegion;
    
    //Find all segmentRectangles that could be a letter along with the current segmentRectangle
    for(int j=0;j<segmentRects.size();j++){
        if(!inATextRegion[j] && j!=current_index && couldBeTwoLetters(segmentRects[current_index].rect, segmentRects[j].rect, segmentRects[current_index].color, segmentRects[j].color)){
            inATextRegion[j] = true;
            lettersInMyRegionIndexes.push_back(j);
            rectangle(img, segmentRects[j].rect.tl(), segmentRects[j].rect.br(), newVal);
        }
    }
 
    //Add all letters associated with the current letter's associates to the text region as well
    for(int k=0;k<lettersInMyRegionIndexes.size();k++){
        vector<segmentRectangle> s = getLettersInMyRegion(segmentRects,lettersInMyRegionIndexes[k],inATextRegion, img,newVal);
        lettersInMyRegion.insert(lettersInMyRegion.end(), s.begin(), s.end());
        lettersInMyRegion.push_back(segmentRects[lettersInMyRegionIndexes[k]]);
    }

    //Get rid of duplicates
    vector<segmentRectangle> noDups;
    for(int k=0;k<lettersInMyRegion.size();k++){
        bool alreadyPresent = false;
        for(int l=0;l<noDups.size();l++){
            if(lettersInMyRegion[k].rect==noDups[l].rect){
                alreadyPresent = true;break;
            }
        }
        if(!alreadyPresent){
            noDups.push_back(lettersInMyRegion[k]);
        }
    }
    
    return noDups;
}

//This function takes an image and returns a vector of rectangles that correspond to text regions within the image
/*
 The way this function works is that it first finds all segments in the image.
 It then takes all the segmentRectangles (which represent the segments) and for each one sees that it, with another segmentRectangle, could it be one of two Letters.
 Words are grouped together this way and then return.
 */
vector<Rect> getTextRegions(Mat img,vector<vector<Point>> contours,vector<Vec4i> hierarchy,string window_name,Mat ground){
    
    int color_difference = 20;
    int min_rect_width = 5;
    int min_rect_height = 7;
    vector<segmentRectangle> segmentRects = getSegmentsRectanglesWithContours(img,contours,hierarchy,color_difference,min_rect_width,min_rect_height); //Proven to be deterministic (even with order of segmentRectangles returned)
    
    //Create an int array to keep track of the graphs that make up text regions
    int* textRegionIntArray = new int[segmentRects.size()]; //Keeps track of which region each segment is in
    for(int i=0;i<segmentRects.size();i++){
        textRegionIntArray[i] = i;
    }
    vector<Rect> textRegions;
    
    //For every rectangle i in the image , add any rectangle j that could be a letter with it (and any rectangle k in j's region) to i's region
    for(int i=0;i<segmentRects.size();i++){
        int i_region = textRegionIntArray[i]; //Region number
        //Check if any rectangle j could be a letter with rectangle i
        for(int j=0;j<segmentRects.size();j++){
            if(couldBeTwoLetters(segmentRects[i].rect, segmentRects[j].rect, segmentRects[i].color, segmentRects[j].color)){
                int j_region = textRegionIntArray[j];
                //Add every rectangle k, from j's region, to rectangle i's region
                for(int k=0;k<segmentRects.size();k++){
                    if(textRegionIntArray[k]==j_region){ textRegionIntArray[k] = i_region; }
                }
            }
        }
    }
        
    //At this stage we have an array of numbers, each index corresponding to a segmentRect with the number of its text region.
    
    
    
    //For all text region numbers, get the enclosing rectangle of the text region
     RNG rng = theRNG();
     /*
    for (int i = 0;i<segmentRects.size();i++) {
        int i_region = i;
        int char_count = 0;
        Scalar newVal( rng(256), rng(256), rng(256) );
        int tlx = 2147483647; int tly = 2147483647; int brx = -1; int bry = -1;
        //Get the maximum tl and br of the text region
        for(int j=0;j<segmentRects.size();j++){
            if(textRegionIntArray[j]==i_region){
                char_count++;
                tlx = min(tlx, segmentRects[j].rect.tl().x);
                tly = min(tly, segmentRects[j].rect.tl().y);
                brx = max(brx, segmentRects[j].rect.br().x);
                bry = max(bry, segmentRects[j].rect.br().y);
            }
        }
        cv::Rect boundingRect(Point(tlx,tly),Point(brx,bry));
        //rectangle(img, boundingRect, newVal);
        if(char_count>=2){
            textRegions.push_back(boundingRect);
        }
    }
     */
    
    Mat textRegionsImage = img.clone();
    for (int i = 0;i<segmentRects.size();i++) {
        int i_region = i;
        Scalar newVal( rng(256), rng(256), rng(256) );
        for(int j=0;j<segmentRects.size();j++){
            if(textRegionIntArray[j]==i_region){
                rectangle(textRegionsImage,segmentRects[j].rect, newVal);
            }
        }
    }
    
    //Gather intersecting text regions together
    vector<Rect> intersectFinishedTextRegions = textRegions;
    bool anyStillIntersecting = true;
    while(anyStillIntersecting){
        //intersectFinishedTextRegions = encloseIntersectingRects(intersectFinishedTextRegions);
        anyStillIntersecting = false;
        for(int i=0;i<intersectFinishedTextRegions.size();i++){
            for(int j=0;j<intersectFinishedTextRegions.size();j++){
                if(i!=j && ((intersectFinishedTextRegions[i] & intersectFinishedTextRegions[j]).area() > 0) ){ anyStillIntersecting = true; }
            }
        }
    }
    
    return intersectFinishedTextRegions;
}

//This function determines whether two characters should be grouped in the same region of text, by their size, color, and how near they are vertically
bool shouldBeInSameRegionOfText(segmentRectangle c1, segmentRectangle c2){
    
    float allowed_height_ratio = 1.82; //1.8 splits up the s in 'This' and the g in 'grass' on last notice. 1.85 puts together the second and third sign on notice 4 (the blue 3)
    float allowed_width_ratio = 2.5;
    float height_increase_for_overlap = 1.2;
    float width_increase_for_overlap = 0;//3.5;
    float allowed_color_difference = 30;
    
    return areRoughlyTheSameSize(c1.rect,c2.rect,allowed_height_ratio,allowed_width_ratio)
    && areCloseBy(c1.rect, c2.rect,height_increase_for_overlap,width_increase_for_overlap)
    && areTheSameColor(c1.color, c2.color,allowed_color_difference);
}


//This function determines whether two lines of text are in the same text region
bool areInTheSameTextRegion(vector<segmentRectangle> l1, vector<segmentRectangle> l2){
    for(int i=0;i<l1.size();i++){
        for(int j=0;j<l2.size();j++){
            //Return true if any two characters from the separate lines of text should be in the same region as eachother (They are of similar size / color / are nearby vertically)
            if(shouldBeInSameRegionOfText(l1[i],l2[j])){
                return true;
            }
        }
    }
    return false;
}

//This function takes an image, its contours and hierarchy and returns a vector of lines of text. Each line of text is a vector of segmentRectangles, which correspond to a rectangles in the image where characters have been detected.
/*
 The way this function works is that it first finds all segments in the image.
 It then takes all the segmentRectangles (which represent the segments) forms them into groups, on the criteria that they 'could be letters' together.
 */
vector<vector<segmentRectangle>> getLinesOfText(Mat img,vector<vector<Point>> contours,vector<Vec4i> hierarchy,string window_name){
    
    //Get the rectangle surrounding every component in the image and its color.
    int color_difference = 20;
    int min_rect_width = 5;
    int min_rect_height = 7;
    vector<segmentRectangle> segmentRects = getSegmentsRectanglesWithContours(img,contours,hierarchy,color_difference,min_rect_width,min_rect_height); //Proven to be deterministic (even with order of segmentRectangles returned)
    
    //Create an int array to keep track of the graphs that make up lines of text
    int* lineIntArray = new int[segmentRects.size()]; //Keeps track of which line of text each component is in
    for(int i=0;i<segmentRects.size();i++){
        lineIntArray[i] = i;
    }
    vector<vector<segmentRectangle>> linesOfText;
    
    //For every componenent i in the image , add any component j that 'could be a letter' with it (and any component k in j's line of text) to i's line of text
    for(int i=0;i<segmentRects.size();i++){
        int i_line = lineIntArray[i]; //Region number
        //Check if any rectangle j could be a letter with rectangle i
        for(int j=0;j<segmentRects.size();j++){
            if(couldBeTwoLetters(segmentRects[i].rect, segmentRects[j].rect, segmentRects[i].color, segmentRects[j].color)){
                int j_line = lineIntArray[j];
                //Add every component k, from j's line of text, to i's line of text
                for(int k=0;k<segmentRects.size();k++){
                    if(lineIntArray[k]==j_line){ lineIntArray[k] = i_line; }
                }
            }
        }
    }
    
    //At this stage we have an array of numbers,  with each index corresponding to a segmentRect and the number of the line of text it is in.
    
    //For all lines of text, add them to the vector of lines of text to return
     for (int i = 0;i<segmentRects.size();i++) {
         
         vector<segmentRectangle> current_line;
         
         //If the current component's line of text hasn't already been added, then add it
         if(lineIntArray[i]>=0){
             int i_line = i;
             int char_count = 0;
             //Count up the components in the line. Each line of text is only returned if it has at least two components in it. (otherwise it isn't a line of text)
             for(int j=0;j<segmentRects.size();j++){
                 if(lineIntArray[j]==i_line){
                     char_count++;
                     current_line.push_back(segmentRects[j]);
                     lineIntArray[j] = -1;
                 }
             }
             if(char_count>=2){
                 linesOfText.push_back(current_line);
             }
         }
     }
    
    return linesOfText;

}

//This function takes a vector of lines of text in the form vector<vector<segmentRectangle>> and returns a vector of text regions. Each text region is a vector of segmentRectangles, which correspond to the rectangles in the image where characters have been detected.
vector<vector<segmentRectangle>> joinLinesOfText(vector<vector<segmentRectangle>> linesOfText){
    
    //Create an int array to keep track of the graphs that make up text regions
    int* textRegionIntArray = new int[linesOfText.size()]; //Keeps track of which text region each line-of-text is ine
    for(int i=0;i<linesOfText.size();i++){
        textRegionIntArray[i] = i;
    }
    
    //For every line-of-text i in the image , add any line-of-text j that is in the same text region
    for(int i=0;i<linesOfText.size();i++){
        int i_region = textRegionIntArray[i]; //Region number
        //Check if any line-of-text j is part of the same text region as line-of-text i
        for(int j=0;j<linesOfText.size();j++){
            if(areInTheSameTextRegion(linesOfText[i],linesOfText[j])){
                int j_region = textRegionIntArray[j];
                //Add every line-of-text k from j's region, to line-of-text i's region
                for(int k=0;k<linesOfText.size();k++){
                    if(textRegionIntArray[k]==j_region){ textRegionIntArray[k] = i_region; }
                }
            }
        }
    }
    
    vector<vector<segmentRectangle>> textRegions;
    //For all text regions, add them to the vector of text regions to return
    for (int i = 0;i<linesOfText.size();i++) {
        //If the current line-of-text's text region hasn't already been added, then add it
        if(textRegionIntArray[i]>=0){
            vector<segmentRectangle> current_region;
            int i_region = i;
            //Add any line of text from this same text region to the current_region
            for(int j=0;j<linesOfText.size();j++){
                if(textRegionIntArray[j]==i_region){
                    //Add all characters in this line of text to the text region
                    for(int k=0;k<linesOfText[j].size();k++){
                        current_region.push_back(linesOfText[j][k]);
                    }
                    //Note that this line of text has been added to a text region
                    textRegionIntArray[j] = -1;
                }
            }
            if(current_region.size()>0){textRegions.push_back(current_region);}
        }
    }
    return textRegions;
}

//This function takes a vector of text regions and returns a vector or rectangles that correspond to the bounding rectangle of each text-region
vector<Rect> getTextRegionRectangles(vector<vector<segmentRectangle>> textRegions,int min_region_width = 0, int min_region_height = 0){
    
    vector<Rect> regionRects;
    
    //For all text regions that are connected, get the enclosing rectangle of the text region
    for (int i=0;i<textRegions.size();i++) {
        //Get the maximum tl and br of the text region
        int tlx = 2147483647; int tly = 2147483647; int brx = -1; int bry = -1;
        for(int j=0;j<textRegions[i].size();j++){
            tlx = min(tlx, textRegions[i][j].rect.tl().x);
            tly = min(tly, textRegions[i][j].rect.tl().y);
            brx = max(brx, textRegions[i][j].rect.br().x);
            bry = max(bry, textRegions[i][j].rect.br().y);
        }
        cv::Rect boundingRect(Point(tlx,tly),Point(brx,bry));
        if(boundingRect.width>=min_region_width && boundingRect.height>=min_region_height){
            regionRects.push_back(boundingRect);
        }
    }
    
    //Gather intersecting text regions together
    vector<Rect> intersectFinishedTextRegions = regionRects;
    bool anyStillIntersecting = true;
    while(anyStillIntersecting){
        intersectFinishedTextRegions = encloseIntersectingRects(intersectFinishedTextRegions);
        anyStillIntersecting = false;
        for(int i=0;i<intersectFinishedTextRegions.size();i++){
            for(int j=0;j<intersectFinishedTextRegions.size();j++){
                if(i!=j && ((intersectFinishedTextRegions[i] & intersectFinishedTextRegions[j]).area() > 0) ){ anyStillIntersecting = true; }
            }
        }
    }
    
    return intersectFinishedTextRegions;
}

//-----------------------------------------------------------------------------------------------------------------

//Finds the DICE Coefficient, of a notice image. Given the rectangles corresponding to detected text regions and ground truths
//Pre-requistite: No detected text regions must overlap and no ground truths must overlap
float getDiceCoefficient(vector<Rect> trs, vector<Rect> gts){
    
    /*
      "The DICE cooefficient is 2 times the Area of Overlap
     (between the ground truth and the regions found by your program)
     divided by the sum of the Area of the Ground Truth and the Area of the regions found by your program." - Ken
     */
    
    //Check that no trs or gts overlap
    for(int i=0;i<trs.size();i++){
        for(int j=i+1;j<trs.size();j++){
            assert((trs[i] & trs[j]).area() <= 0);
        }
    }
    for(int i=0;i<gts.size();i++){
        for(int j=i+1;j<gts.size();j++){
            assert((gts[i] & gts[j]).area() == 0);
        }
    }
    
    //Total overlapping area of the detected text regions and the ground truths
    float total_overlap_area = 0;
    for(int i=0;i<trs.size();i++){
        for(int j=0;j<gts.size();j++){
            total_overlap_area += (trs[i] & gts[j]).area();
        }
    }
    
    //Total area of the ground truths
    float ground_truths_area = 0;
    for(int i=0;i<gts.size();i++){
        ground_truths_area+=gts[i].area();
    }
    
    //Total area of the detected text regions
    float text_regions_area = 0;
    for(int i=0;i<trs.size();i++){
        text_regions_area+=trs[i].area();
    }
    
    return (2*total_overlap_area) / (text_regions_area + ground_truths_area);
}

//This function finds the regions of text within an image (under development and doesn't do that at the moment)
vector<Rect> find_text(string window_name,cv::Mat img){
    
    //Perform mean shift segmentation on the image
    int spatial_radius = 80;
    int color_radius = 50;
    int maximum_pyramid_level = 0;
    cout << "here" << endl;
    Mat meanS = meanShiftSegmentation(img, spatial_radius, color_radius, maximum_pyramid_level);
    cout << "here now" << endl;
    
    //Flood fill the image
    int flood_fill_color_difference = 5;
    Mat floodFillImage = floodFill(img, flood_fill_color_difference);

    //Convert the image to grayscale
    Mat grayscaleMat = grayscale(floodFillImage);

    //Convert the grayscale image into a binary image
    int block_size = 19;
    int offset = 20;
    int output_value = 255;
    Mat binaryMat = binary(grayscaleMat,block_size,offset,output_value);

    //Find the contours within the image (Connected Components Analysis)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryMat,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //Draw the contours of the image
    Mat connected = fillContours(img, contours);
    
    vector<vector<segmentRectangle>> linesOfText = getLinesOfText(img, contours, hierarchy, window_name);
    
    vector<vector<segmentRectangle>> textRegions = joinLinesOfText(linesOfText);
    
    vector<Rect> textRegionsRectangles = getTextRegionRectangles(textRegions);
    
    RNG rng = theRNG();
    
    /* Just an example of what getLinesOfText returns */
    Mat linesImg = img.clone();
    for(int i=0;i<linesOfText.size();i++){
        Scalar newVal( rng(256), rng(256), rng(256) );
        for(int j=0;j<linesOfText[i].size();j++){
            rectangle(linesImg, linesOfText[i][j].rect, newVal);
        }
    }
    
    /* Just an example of what joinLinesOfText returns */
    Mat joinsImg = img.clone();
    for(int i=0;i<textRegions.size();i++){
        Scalar newVal( rng(256), rng(256), rng(256) );
        for(int j=0;j<textRegions[i].size();j++){
            rectangle(joinsImg, textRegions[i][j].rect, newVal,2);
        }
    }
    
    /* Just an example of what getRegionRectangles returns*/
    Mat boundsImg = img.clone();
    for(int i=0;i<textRegionsRectangles.size();i++){
        rectangle(boundsImg, textRegionsRectangles[i], Scalar(0,0,255));
    }
    
    Mat first[] = {img,floodFillImage};
    Mat second[] = {grayscaleMat,binaryMat};
    Mat third[] = {boundsImg};
    display_images("third",2,third);
    waitKey(0);
    
    return textRegionsRectangles;
}


//------------------------------------------------ MAIN PROGRAM --------------------------------------------------------

int main(int argc, char** argv){
    
    //Load images from their directory
    string path_to_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/notice-images/";
    string path_to_grounds = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/notice-ground-truth-images/";
    string path_to_ground_truths_json = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/text-positions.json";
    string path_to_shifts = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/mean-shifts/";
    const int number_of_images = 14;
    Mat images[number_of_images];
    Mat grounds[number_of_images];
    vector<vector<Rect>> ground_truths = read_ground_truths(path_to_ground_truths_json);
    float dice_coeffs[number_of_images];
    
    //Process each image
    for(int i=9;i<14;i++){
        
        //Read in the image
        string image_name = "Notice"+to_string(i+1)+".jpg";
        //string mean_name = "Notice"+to_string(i)+"-mpl50.jpg";
        images[i] = imread(path_to_images+image_name);
        grounds[0] = imread(path_to_grounds+"Notice7.jpg");
        if(images[i].empty()){ return -1; }
        cout << "1738" << endl;
        vector<Rect> detected_text_regions = find_text("Notice"+to_string(i),images[i]);
        //dice_coeffs[i] = getDiceCoefficient(detected_text_regions, ground_truths[i]);
    }
    
    float total_dice = 0;
    for(int i=0;i<number_of_images;i++){
        total_dice += dice_coeffs[i];
        cout << "Notice"+to_string(i)+" DICE: "+to_string(dice_coeffs[i]) << endl;
    }
    
    cout << "Average DICE Coefficient: "+to_string( total_dice / number_of_images ) << endl;
    
    return 0;
}
