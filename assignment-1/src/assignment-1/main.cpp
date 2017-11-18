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

//---------------------- VISION TECHNIQUES ---------------------------

//This function Grayscales an image
Mat grayscale(Mat img){
    Mat grayscaleMat;
    cvtColor(img, grayscaleMat, CV_RGB2GRAY);
    return grayscaleMat;
}

//This function turns a grayscale image into a binary image using apaptiveThresholding
Mat apaptiveBinary(Mat grayscaleImage,int block_size,int offset, int output_value){
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
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            if( mask.at<uchar>(y+1, x+1) == 0 ){
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, Scalar::all(color_difference), Scalar::all(color_difference),4);
            }
        }
    }
    return img;
}

//This function takes an image and mean-shift parameters and returns a version of the image that has had mean shift segmentation performed on it
Mat meanShiftSegmentation( Mat img, int spatial_radius, int color_radius, int maximum_pyramid_level ){
    Mat res = img.clone();
    pyrMeanShiftFiltering(img,res,spatial_radius,color_radius,maximum_pyramid_level);
    return res;
}

//-------------------------------------------FUNCTIONS TO MAKE USE OF------------------------------------------------------

class segmentRectangle{
    public:
        Scalar color;
        Rect rect;
        //Overload == operator for segmentRectangle
        friend bool operator==(const segmentRectangle &lhs, const segmentRectangle &rhs){
            return (lhs.color==rhs.color && lhs.rect==rhs.rect);
        }
};


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

//This function takes an image & its contours and fills the contours with random colors
Mat fillContours(Mat img, vector<vector<Point>> contours){
    Mat contoursMat = img;
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Scalar colour( (rand()&255), (rand()&255), (rand()&255) );
            drawContours( contoursMat, contours, i, colour,CV_FILLED);
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


//-------------------------- FINDING TEXT SPECIFICALLY ----------------------------------

//This function returns true if the second rectangle is enclosed within the first
bool isEnclosing(Rect r1, Rect r2){
    return ( r2.tl().x > r1.tl().x && r2.tl().y > r1.tl().y && r2.br().x < r1.br().x && r2.br().y < r2.br().y );
}

//This function returns true if the two rectangles intersect
bool intersect(Rect r1, Rect r2){
    return ((r1 & r2).area() > 0);
}

//This function checks that two rectangles are close by to eachother
bool areCloseBy(Rect r1, Rect r2,float height_increase_for_overlap,float width_increase_for_overlap){
    Point newTL(r1.tl().x-(width_increase_for_overlap*(r1.width/2)),r1.tl().y-(height_increase_for_overlap*(r1.height/2)));
    Point newBR(r1.br().x+(width_increase_for_overlap*(r1.width/2)),r1.br().y+(height_increase_for_overlap*(r1.height/2)));
    Rect r3(newTL,newBR);
    return intersect(r3, r2);
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
    
    float allowed_height_ratio = 2.5;
    float allowed_width_ratio = 2.5;
    float height_increase_for_overlap = 2;
    float width_increase_for_overlap = 3.5;
    float allowed_color_difference = 40;
    
    return areRoughlyTheSameSize(r1,r2,allowed_height_ratio,allowed_width_ratio)
                && areCloseBy(r1, r2,height_increase_for_overlap,width_increase_for_overlap)
                    && areTheSameColor(r1_color, r2_color,allowed_color_difference);
}


//
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
 It then takes all the segmentRectangles (which represent the segments) and for each one, sees that with another segmentRectangle, could it be one of Two Letters.
 Words are grouped together this way and then return.
 */
vector<Rect> getTextRegionsWithRecursion(Mat img){
    
    int color_difference = 20;
    int min_rect_width = 5;
    int min_rect_height = 7;
    vector<segmentRectangle> segmentRects = getSegmentsRectangles(img, color_difference,min_rect_width,min_rect_height); //Proven to be deterministic (even with order of segmentRectangles returned)
    
    bool* inATextRegion = new bool[segmentRects.size()]; //Keeps track of which segments are already in text regions
    
    vector<Rect> textRegions;
    
    RNG rng = theRNG();
    for(int i=0;i<segmentRects.size();i++){
        
        //If letter is already in a text region, then move on to the next letter
        if(inATextRegion[i]){continue;}
        
        //Find all letters in this letter's text region
        Scalar newVal( rng(256), rng(256), rng(256) );
        vector<segmentRectangle> textRegion = getLettersInMyRegion(segmentRects, i,inATextRegion,img,newVal);
        textRegion.push_back(segmentRects[i]); //Put the current letter into the text region as well
        
        //Mark these letters as in text regions
        /* Not sure I need this -> maybe they are all already markedd as in text regions at this stage?
         for(int j=0;j<textRegion.size();j++){
         for(int k=0;k<segmentRects.size();k++){
         if(textRegion[j]==segmentRects[k]){
         inATextRegion[k] = true;
         }
         }
         }
         */
        
        //Add the enclosing rectangle of this text region to the textRegions
        int tlx = 2147483647; int tly = 2147483647; int brx = -1; int bry = -1;
        for (int i = 1;i<textRegion.size();i++) {
            tlx = min(tlx, textRegion[i].rect.tl().x);
            tly = min(tly, textRegion[i].rect.tl().y);
            brx = max(brx, textRegion[i].rect.br().x);
            bry = max(bry, textRegion[i].rect.br().y);
        }
        cv::Rect boundingRect(Point(tlx,tly),Point(brx,bry));
        textRegions.push_back(boundingRect);
    }
    
    //TODO add code here that puts all rectangles that interesect together before returning the text regions.
    
    return textRegions;
}

//This function takes an image and returns a vector of rectangles that correspond to text regions within the image
/*
 The way this function works is that it first finds all segments in the image.
 It then takes all the segmentRectangles (which represent the segments) and for each one, sees that with another segmentRectangle, could it be one of Two Letters.
 Words are grouped together this way and then return.
 */
vector<Rect> getTextRegionsNoRecursion(Mat img){
    
    int color_difference = 20;
    int min_rect_width = 5;
    int min_rect_height = 7;
    vector<segmentRectangle> segmentRects = getSegmentsRectangles(img, color_difference,min_rect_width,min_rect_height); //Proven to be deterministic (even with order of segmentRectangles returned)
    
    //Create an int array to keep track of the graphs that make up text regions
    int* textRegionIntArray = new int[segmentRects.size()]; //Keeps track of which region each segment is in
    for(int i=0;i<segmentRects.size();i++){
        textRegionIntArray[i] = i;
    }
    
    vector<Rect> textRegions;
    
    RNG rng = theRNG();
    //For every rectangle i in the image , add any rectangle j that could be a letter with it (and any rectangle k in j's region) to i's region
    for(int i=0;i<segmentRects.size();i++){
        
        int i_region = textRegionIntArray[i]; //Region number
        
        //Check if any rectangle k could be a letter with rectangle i
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
    
    
    //TODO add code here that puts all rectangles that interesect together before returning the text regions.
    
    return textRegions;
}

//-------------------------------------------------------------------------------------------

//This function finds the regions of text within an image (under development and doesn't do that at the moment)
void find_text(string window_name,cv::Mat img){
    
    //Declare variables
    Mat contours_image, croppedImage;
    
    /*
     //Perform mean shift segmentation on the image
     int spatial_radius = 80;
     int color_radius = 45;
     int maximum_pyramid_level = 0;
     Mat meanShiftSegmentationImage = meanShiftSegmentation(img, spatial_radius, color_radius, maximum_pyramid_level);
     */
    
    //Flood fill the image
    //int color_difference = 20;
    //Mat floodFillImage = floodFill(img, color_difference);
    
    
    /*
     //Convert the image to grayscale
     Mat grayscaleMat = grayscale(img);
     
     //Convert the grayscale image into a binary image
     int block_size = 19;
     int offset = 20;
     int output_value = 255;
     Mat binaryMat = apaptiveBinary(grayscaleMat,block_size,offset,output_value);
     
     
     
     //Close and Open the image
     int n = 2;
     Mat closedMat = close(binaryMat,n);
     Mat openedMat = open(binaryMat,n);
     
     //Find the contours within the image (Connected Components Analysis)
     vector<vector<Point>> contours;
     vector<Vec4i> hierarchy;
     //findContours(binaryMat,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
     
     //Find and draw rectangles around the contours with more than 4 children
     //vector<vector<Point>> contoursWithNChildren = getContoursWithNChildren(contours, hierarchy, 4);
     
     //Draw the contours of the image
     //Mat signsImg = fillContours(img, contoursWithNChildren);
     */
    
    Mat img01, img02, img03, img1,img2,img3,img4,img5,img6;
    //cv::pyrMeanShiftFiltering (img, img1, sp, sr, 1, TermCriteria(TermCriteria::MAX_ITER, 5, 1));
    Mat vis = img.clone();
    vector<Rect> textRegions = getTextRegionsNoRecursion(img);
    for(int i=0;i<textRegions.size();i++){
        cout << "now adding rectangle "+to_string(i) << endl;
        rectangle(img, textRegions[i].tl(), textRegions[i].br(), Scalar(0,0,255));
    }
    
    Mat poops[] = {img};
    display_images(window_name,1,poops);
    
    //cv::Mat images[] = {meanShiftSegmentationImage,grayscaleMat,binaryMat};
    //display_images(4,images);
    
}


//-------------------------- MAIN PROGRAM ----------------------------------

int main(int argc, char** argv){
    
    //Load images from their directory
    string path_to_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/notice-images/";
    string path_to_shifts = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/mean-shifts/";
    const int number_of_images = 8;
    Mat images[number_of_images];
    
    //Process each image
    for(int i=0;i<number_of_images;i++){
        
        //Read in the image
        string image_name = "Notice"+to_string(i+1)+".jpg";
        string mean_name = "mean-shift-"+to_string(i)+".jpg";
        images[i] = imread(path_to_shifts+mean_name);
        if(images[i].empty()){ return -1; }
        find_text("Notice "+to_string(i),images[i]);
        std::cout << "Processed Notice "+to_string(i) << std::endl;
        waitKey(0);
    }
    return 0;
}
