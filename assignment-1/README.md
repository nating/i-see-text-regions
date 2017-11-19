

# Locating Printed Text on Notices

This report is on an OpenCV program developed to locate text on notices.

## Contents

1. Introduction
2. Overview of Report
3. Initial Thoughts on problem?
4. Overview of Solution
3. Technical Details of Solution < This should include issues/problems (and suggestions for improvements if possible).>
4. Discussion of Results < discussion of the results including any issues/problems with the metrics and the ground truth (and suggestions for improvements if possible) >
8. Overview of Source Code
9. Thanks


## Introduction

The problem is to locate printed text on notices. An OpenCV program was developed to take in images containing notices and locate the regions within the images that contain text.  
The initial plan during development was to locate notices within the images and then locate the regions of text within the notices. This would make it easier to determine what parts of the image were text, as notices are unlikely to contain many components other than their text, and the program would not have to deal with the noise which is more common outside of the notices. The first attempts at designing the program had in mind to identify notices in images, but no technique was found to successfully find the notices in every sample image.  
Instead, later the focus of the program became to locate text within the images under the assumption that the text will take up a reasonable portion of the images.  

## Overview of the Report

This report details the program developed for locating text on notices, the problems developing it, how the program works, and how well the program performs. It includes

## Overview of the Solution

The steps involved in the program are:
* Read in image
* Perform Mean Shift Segmentation on the image
* Convert the image to Grayscale
* Convert the grayscale image to a binary one with OTSU thresholding
* Do connected components components analysis on the image
* For every component in the image of a certain size, I find the average color of the component in the original image
*



#################### NOTES FOR MAKING SOLUTION BETTER

* Text regions must be of a certain height and width in terms of pixels for them to count (gets rid of top of notice 2, top of notice 5)

## What to implement

components = getComponents(img);

linesOfText = findLinesOfText(img,components); //Checks which components 'couldBeLetters' with each other horizontally.

textRegions = joinLinesOfText(img,linesOfText); //Checks if characters within lines of text 'couldBeLetters' (using height) with other lines of text, and if so, joins them.



'linesOfText' should be a vector<vector<SegmentRect>> which is a list of lists of characters ( a list of lines of text )


















/
