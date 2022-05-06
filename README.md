# NLP-Project-4

## Overview

This is the README file for Project 4 of Dr. Michael Bloodgood's Natural Language Processing (CSC 427-01) class at The College of New Jersey, completed by Michael Giordano, Rebecca Goldberg, Casey Lishko, and Poean Lu. 

## Contents

Our package includes the source code scripts for the program, as well as training sets and the testMaster.txt file with text from a sentiment labelled dataset. \
Corpus: Sentiment Labelled Sentences Dataset from the UCI Machine Learning Repository


Source files in folder:
- createDataFiles.py - creates the various test and training files used in our program
- main.py - a script containing the code to implement T3-T9
- plot1.png - our graphical output of plot 1 for T9
- plot2.png - our graphical output of plot 2 for T9
- D6.txt - a text file providing our answers to questions asked for D6
- README.md - this file, explains the contents to the user.

## Requirements

Operating System: Ubuntu 
Language: Python 3.7.5 

## Installation

In the ELSA command line run:
$ module add python/3.7.5

## Use

This program is designed to be used from the terminal. Once the user has entered the directory where they have unzipped the tar.gz file, they can run the python programs.

To run createDataFiles.py, the user will enter: \
$ python3 createDataFiles.py ****

To run main.py, the user will enter: \
$ python3 main.py ****



## Notes
Normalization \
data was converted to all lowercase letters \
all punctuation in the data set was removed \

We used B = 1000 for our testing and plot output

Output: \
the program will output: \
- plot1.png - as a new file
- plot2.png - as a new file
