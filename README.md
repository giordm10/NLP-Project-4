# NLP-Project-4

## Overview

This is the README file for Project 4 of Dr. Michael Bloodgood's Natural Language Processing (CSC 427-01) class at The College of New Jersey, completed by Michael Giordano, Rebecca Goldberg, Casey Lishko, and Poean Lu. 

## Contents

Our package includes the source code scripts for the program, as well as training sets and the testMaster.txt file with text from a sentiment labelled dataset.
Corpus: Sentiment Labelled Sentences Dataset from the UCI Machine Learning Repository


Source files in folder:
- main.py - a script containing all of our source code
- plot1.png - our graphical output of plot 1 for T9
- plot2.png - our graphical output of plot 2 for T9
- d3_output.txt - a text file containing all of the ordered pairs of (deltaF1, pValue) 
- D5.txt - a text file containing our analysis for T10
- D6.txt - a text file providing our answers to questions asked for D6
- README.md - this file explains the contents of our package to the user.
- sentiment labelled sentences - directory that contains amazon_cells_labelled.txt, imdb_labelled.txt, yelp_labelled.txt, three text files that we concatenated into fulldataLabeled.txt
- trainMaster.txt - a text file containing lines corresponding to randomly generated line numbers from fulldataLabeled.txt (T2)
- testMaster.txt - a text file containing the lines that did not end up in trainMaster.txt file (T2)
- fulldataLabeled.txt - a long text file concatenated from the three files in the "sentiment labelled sentences" directory, whose lines we split into train and test files
- trainingSets - a folder containing the subdirectories for all of the smaller training sets (T3)

## Requirements

Operating System: Ubuntu 
Language: Python 3.7.5 

## Installation

In the ELSA command line run:
$ module add python/3.7.5

Matplotlib was used to create the graph. Install that library by running the following command:
$ pip install matplotlib

## Use

This program is designed to be used from the terminal. Once the user has entered the directory where they have unzippedped the tar.gz file, they can run the Python programs.

To run main.py, the user will enter:  
$ python3 main.py /path/to/fulldataLabeled.txt /path/to/trainMaster.txt /path/to/testMaster.txt

where /path/to/fulldataLabeled.txt is the path to the fulldataLabeled.txt file, /path/to/trainMaster.txt is the path to the trainMaster.txt file, and /path/to/testMaster.txt is the path to testMaster.txt

## Notes
### Normalization
All text in fulldataLabeled.txt was lowercased and stripped of punctuation

We used b = 1000 as the number of bootstrap samples (test files whose 400 lines were randomly chosen from the original testMaster.txt with replacement)

Runtime: When we tested on our program on the ELSA HPC Cluster, it took about 30 minutes to run

### Output:
The program will output:  
- d3_output.txt - a file containing each pair of 1770 (deltaF1, pValue) ordered pairs
- plot1.png - shows the first plot for T9
- plot2.png - shows the second plot for T9
