/*
  main.cpp
  assignment-1

  The purpose of this file is to take in images of notices, and to find the positions of text within the notices.
  https://github.com/nating/visionwork/blob/master/assignment-1/src/assignment-1/main.cpp
 
  Created by Geoffrey Natin on 25/10/2017.
  Copyright © 2017 nating. All rights reserved.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "json.hpp" //https://github.com/nlohmann/json
#include <fstream>
#include "vision-techniques.h"
#include "extra-functions.h"
#include "finding-text.h"

using json = nlohmann::json;
using namespace cv;
using namespace std;

//-------------------------------------------------- FUNCTIONS ---------------------------------------------------------

//This function creates images depicting the ground truths for notices passed in a JSON in the format of the JSON that should come packaged with this project. (It's a very specific function!)
void create_ground_truth_images(string path_to_images,string path_to_ground_truth_images,json j){
    for (json::iterator it = j["notices"].begin(); it != j["notices"].end(); ++it) {
        string name = it.value()["name"];
        cv::Mat img = imread(path_to_images+name);
        for (json::iterator z = it.value()["boxes"].begin(); z != it.value()["boxes"].end(); ++z) {
            Point p0 = Point(z.value()[0][0],z.value()[0][1]);
            Point p1 = Point(z.value()[1][0],z.value()[1][1]);
            Rect r = Rect(p0,p1);
            rectangle(img,r,Scalar(0,0,255),2);
        }
        imwrite(path_to_ground_truth_images+name, img);
    }
}

//This function reads ground truths for notice images from a json and returns a vector of ground truths of each image. (each image's ground truths are a vector of rectangles) (It's a very specific function!)
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

//------------------------------------------------ MAIN PROGRAM --------------------------------------------------------

int main(int argc, char** argv){
    
    //Define filepaths
    string path_to_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/images/";
    string path_to_ground_truths = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/ground-truths/";
    string path_to_ground_truths_json = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/text-positions.json";
    string path_to_shifts = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-1/assets/mean-shifts/";
    
    //Initialise variables
    const int number_of_images = 8;
    Mat images[number_of_images];
    float dice_coeffs[number_of_images];
    
    //Read in ground truths
    vector<vector<Rect>> ground_truths = read_ground_truths(path_to_ground_truths_json);
    
    //Process each image
    for(int i=0;i<number_of_images;i++){
        
        //Read in the image
        string image_name = "Notice"+to_string(i+1)+".jpg";
        string mean_name = "Notice"+to_string(i)+".jpg";
        images[i] = imread(path_to_shifts+mean_name);
        if(images[i].empty()){ cout << "Image "+to_string(i)+" empty. Ending program." << endl; return -1; }
        
        //Find text regions and get DICE coefficient
        vector<Rect> detected_text_regions = find_text(i,images[i]);
        dice_coeffs[i] = getDiceCoefficient(detected_text_regions, ground_truths[i]);
    }
    
    //Calculate DICE coefficients
    float total_dice = 0;
    for(int i=0;i<number_of_images;i++){
        total_dice += dice_coeffs[i];
        cout << "Notice"+to_string(i)+" DICE: "+to_string(dice_coeffs[i]) << endl;
    }
    cout << "Average DICE Coefficient: "+to_string( total_dice / number_of_images ) << endl;
    
    return 0;
}
