//
//  finding-text.h
//  assignment-1
//
//  The functions in this file support finding text from images.
//  https://github.com/nating/visionwork/blob/master/assignment-1/src/assignment-1/finding-text.h
//
//  Created by Geoffrey Natin on 20/11/2017.
//  Copyright © 2017 nating. All rights reserved.
//

#ifndef finding_text_h
#define finding_text_h

//------------------------------------------------ FINDING TEXT SPECIFICALLY --------------------------------------------------------

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
bool areTheSameSize(Rect r1, Rect r2,float allowed_height_ratio, float allowed_width_ratio){
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
bool couldBeTwoLetters(Rect r1, Rect r2,Scalar r1_color,Scalar r2_color, float allowed_height_ratio, float allowed_width_ratio, float height_increase_for_overlap,float width_increase_for_overlap, float allowed_color_difference){
    
    return areTheSameSize(r1,r2,allowed_height_ratio,allowed_width_ratio)
    && areCloseBy(r1, r2,height_increase_for_overlap,width_increase_for_overlap)
    && areTheSameColor(r1_color, r2_color,allowed_color_difference);
}

//This function takes an image, its contours and hierarchy and returns vector of segmentRectangles within the image that are at least min_rect_width wide and min_rect_height tall.
//(A segmentRectangle represents a rectangle surrounding a segment in the image, and the average pixel value of the pixels in the segment)
//See inside of function for explanation about 'max_children_for_specific_contour_mask' parameter
vector<segmentRectangle> getSegmentsRectanglesWithContours( Mat img, vector<vector<Point>> contours, vector<Vec4i> hierarchy, int min_rect_width, int min_rect_height,int max_children_for_specific_contour_mask=4){
    
    vector<segmentRectangle> segmentRectangles;
    CV_Assert( !img.empty() );
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    if ( !contours.empty() ) {
        
        //For every contour, create a segmentRectangle
        for ( int i=0; i<contours.size(); i++ ) {
            
            //Create mask to find average pixel value of the contour in the original image
            Mat labels = cv::Mat::zeros(img.size(), CV_8UC1);
            drawContours(labels, contours, i, Scalar(255),CV_FILLED);
            
            /*
             A Problem here is that the mask allows everything inside the outline of the contour *including the contour's children* because we have filled them with 255 in the mask.
             It should not allow through the contour's children. To fix this problem, we fill all the children of the contour with 0 in the mask.
             It can be computationally expensive to fill all the contours children if there are a lot, so we do not fill them if there are more than max_children_for_specific_contour_mask.
             */
            
            //Navigate the children in the hierarchy and fill them black in the mask.
            if (!hierarchy.empty()) {
                //If the contour has children, fill each one black on the mask
                if(hierarchy[i][2]>0){
                    //Count how many children the contour has
                    int j=hierarchy[i][2];
                    int contour_children = 1;
                    while(hierarchy[j][0]>0 && contour_children<=max_children_for_specific_contour_mask){
                        contour_children++;
                        j = hierarchy[j][0];
                    }
                    //If there are less than or equal to max_children_for_specific_contour_mask children, then fill them with 0 in the mask.
                    if(contour_children<=max_children_for_specific_contour_mask){
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
            
            //If the segment's width is greater than min_rect_width and the the segment's height is greater than min_rect_height, then add it to the segmentRectangles to be returned.
            if(boundingRectOfContour.width >= min_rect_width && boundingRectOfContour.height >= min_rect_height){
                segmentRectangle segmentRect;
                segmentRect.color = average_color;
                segmentRect.rect = boundingRectOfContour;
                segmentRectangles.push_back(segmentRect);
            }
        }
    }
    return segmentRectangles;
}

//This function takes a vector of rectangles and returns a copy of the vector of rectangles where every overlapping rectangle has been merged
vector<Rect> encloseIntersectingRects(vector<Rect> textRegions){
    
    vector<Rect> boundingRectangles;
    
    //Create graph to represent the text regions
    int textRegionGraph[textRegions.size()];
    for(int i=0;i<textRegions.size();i++){
        textRegionGraph[i] = i;
    }
    
    //Connect graph of intersecting text regions
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
        boundingRectangles.push_back(boundingRect);
        for(int j=0;j<textRegions.size();j++){
            if(textRegionGraph[j]==i_region){ textRegionGraph[j] = -1; }
        }
    }
    
    /*
     When bounding boxes of intersecting rectangles are made, these new bounding rectangles may overlap where they didn't before.
     For this reason, it is important to run this function recursively until that isn't the case.
     */
    //Check if the bounding rectangles just made intersect with eachother
    for(int i=0;i<boundingRectangles.size();i++){
        for(int j=0;j<boundingRectangles.size();j++){
            if(i!=j && ((boundingRectangles[i] & boundingRectangles[j]).area() > 0) ){
                boundingRectangles = encloseIntersectingRects(boundingRectangles);
            }
        }
    }
    
    return boundingRectangles;
}

//This function determines whether two characters should be grouped in the same region of text, by their size, color, and how near they are vertically
bool areInSameRegionOfText(segmentRectangle c1, segmentRectangle c2){
    
    //TODO: Make these values parameters to the function
    float allowed_height_ratio = 1.82;
    float allowed_width_ratio = 2.5;
    float height_increase_for_overlap = 1.2;
    float width_increase_for_overlap = 0;
    float allowed_color_difference = 30;
    
    return areTheSameSize(c1.rect,c2.rect,allowed_height_ratio,allowed_width_ratio)
    && areCloseBy(c1.rect, c2.rect,height_increase_for_overlap,width_increase_for_overlap)
    && areTheSameColor(c1.color, c2.color,allowed_color_difference);
}

//This function returns true if the two lines of text are in the same text region
bool areInTheSameTextRegion(vector<segmentRectangle> l1, vector<segmentRectangle> l2){
    for(int i=0;i<l1.size();i++){
        for(int j=0;j<l2.size();j++){
            //Return true if any two characters from the separate lines of text should be in the same region as eachother (They are of similar size / color / are nearby vertically)
            if(areInSameRegionOfText(l1[i],l2[j])){
                return true;
            }
        }
    }
    return false;
}

//This function takes an image, its contours and hierarchy and returns a vector of lines of text. Each line of text is a vector of segmentRectangles, which correspond to a rectangles in the image where characters have been detected.
vector<vector<segmentRectangle>> getLinesOfText(Mat img,vector<vector<Point>> contours,vector<Vec4i> hierarchy){
    
    //Get the rectangle surrounding every component in the image and its color.
    int min_rect_width = 5;
    int min_rect_height = 7;
    vector<segmentRectangle> segmentRects = getSegmentsRectanglesWithContours(img,contours,hierarchy,min_rect_width,min_rect_height);
    
    //Create a graph to keep track of chararacters that in the same lines of text
    int* lineIntArray = new int[segmentRects.size()]; //Keeps track of which line of text each component is in
    for(int i=0;i<segmentRects.size();i++){
        lineIntArray[i] = i;
    }
    vector<vector<segmentRectangle>> linesOfText;
    
    //highscore: 0.916105
    float allowed_height_ratio = 1.82; // 1.8 splits up the s in 'This' and the g in 'grass' on last notice. 1.85 puts together the second and third sign on notice 4 (the blue 3)
    float allowed_width_ratio = 2.5;
    float height_increase_for_overlap = 0;//1.5;
    float width_increase_for_overlap = 9;//3.5;
    float allowed_color_difference = 30;
    int minimum_char_count_for_a_line_of_text = 2;
    
    //For every componenent i in the image , add any component j that 'could be a letter' with it (and any component k in j's line of text) to i's line of text
    for(int i=0;i<segmentRects.size();i++){
        int i_line = lineIntArray[i]; //Region number
        //Check if any rectangle j could be a letter with rectangle i
        for(int j=0;j<segmentRects.size();j++){
            if(couldBeTwoLetters(segmentRects[i].rect, segmentRects[j].rect, segmentRects[i].color, segmentRects[j].color,allowed_height_ratio, allowed_width_ratio,height_increase_for_overlap,width_increase_for_overlap,allowed_color_difference)){
                int j_line = lineIntArray[j];
                //Add every component k, from j's line of text, to i's line of text
                for(int k=0;k<segmentRects.size();k++){
                    if(lineIntArray[k]==j_line){ lineIntArray[k] = i_line; }
                }
            }
        }
    }
    
    //At this stage we have an array of numbers,  with each index corresponding to a segmentRect and the number of the line of text it is in.
    
    //For every line of text, add it to the vector of lines of text to return
    for (int i = 0;i<segmentRects.size();i++) {
        vector<segmentRectangle> current_line;
        //If the current component's line of text hasn't already been added, then add it
        if(lineIntArray[i]>=0){
            int i_line = i;
            int char_count = 0;
            //Count up the components in the line, and add them to the line's vector.
            //Each line of text is only returned if it has at least two components in it. (otherwise it isn't a line of text)
            for(int j=0;j<segmentRects.size();j++){
                if(lineIntArray[j]==i_line){
                    char_count++;
                    current_line.push_back(segmentRects[j]);
                    lineIntArray[j] = -1;
                }
            }
            if(char_count>=minimum_char_count_for_a_line_of_text){
                linesOfText.push_back(current_line);
            }
        }
    }
    
    return linesOfText;
}

//This function takes a vector of lines of text in the form vector<vector<segmentRectangle>> and returns a vector of text regions. Each text region is a vector of segmentRectangles, which correspond to the rectangles in the image where characters have been detected.
vector<vector<segmentRectangle>> joinLinesOfText(vector<vector<segmentRectangle>> linesOfText){
    
    //Create a graph to keep track of which lines of text are from what region
    int* textRegionIntArray = new int[linesOfText.size()];
    for(int i=0;i<linesOfText.size();i++){
        textRegionIntArray[i] = i;
    }
    
    //For every line-of-text i in the image, add any line-of-text j that is in the same text region (and any line-of-text k in j's text region) to i's text region
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
    
    //For all text regions that are connected, get the enclosing rectangle of the text region by finding the top left character's tl and the bottom right character's br
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
    
    return encloseIntersectingRects(regionRects);
}

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

//This function returns a vector rectangles represent the bounding rectangles of the text regions within the image
vector<Rect> find_text(int i,cv::Mat img){
    
    /*
     //Perform mean shift segmentation on the image
     int spatial_radius = 80;
     int color_radius = 55;
     int maximum_pyramid_level = 0;
     Mat meanS = meanShiftSegmentation(img, spatial_radius, color_radius, maximum_pyramid_level);
     */
    
    //Convert the image to grayscale
    Mat grayscaleMat = grayscale(img);
    
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
    Mat connected = fillContoursRandom(img, contours);
    
    vector<vector<segmentRectangle>> linesOfText = getLinesOfText(img, contours, hierarchy);
    
    vector<vector<segmentRectangle>> textRegions = joinLinesOfText(linesOfText);
    
    vector<Rect> textRegionsRectangles = getTextRegionRectangles(textRegions);
    
    RNG rng = theRNG();
    
    /* Just an example of what getLinesOfText returns */
    Mat linesImg = img.clone();
    for(int i=0;i<linesOfText.size();i++){
        Scalar newVal( rng(256), rng(256), rng(256) );
        for(int j=0;j<linesOfText[i].size();j++){
            rectangle(linesImg, linesOfText[i][j].rect, newVal,2);
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
        rectangle(boundsImg, textRegionsRectangles[i], Scalar(0,0,255),2);
    }
    
    vector<Mat> bandw = {grayscaleMat,binaryMat};
    vector<Mat> color = {boundsImg};
    //display_images("bandw",bandw);
    //display_images("Color",color);
    //waitKey(0);
    
    return textRegionsRectangles;
}

#endif /* finding_text_h */
