from gettext import translation
from random import randint
import linecache as lc
import os
import string
from numpy import full

#T2 part one
def creatOneFile():
    #  List of all files we are tryng to get data from
    original_files = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
    file_paths = []

    # TODO: make this user input
    dir = "/home/hpc/giordm10/CSC427/NLP-Project-4/sentiment labelled sentences"

    # Creates a list of all full path files
    for file in original_files:
        file_paths.append(os.path.join(dir, file))

    # Open fulldataLabeled.txt in write mode
    with open('fulldataLabeled.txt', 'w') as outfile:

        # Iterate through list of all file names
        for names in file_paths:

            # Open each file in read mode
            with open(names) as infile:
                
                # take the information from input files and compile them all together into one file
                outfile.write(infile.read())

def normalize():
    text_doc = []
    #normalize texts
    with open("fulldataLabeled.txt") as f:
        for line in f.readlines():
                # removes punctuation
                line = line.translate(str.maketrans('', '', string.punctuation))
                # makes everything lowercase
                line = line.lower()
                sentence = line.split()

                text_doc.append(sentence)

    #write information back into a file
    with open("fulldataLabeled.txt", "w") as filehandle:
        for review in text_doc:
            for word in review:
                filehandle.write("%s " % word)
            filehandle.write("\n")

#T2 part two
def trainTestSplit():
    testNumbers = []
    count = 0
    #Creates list of test numbers
    while count < 2600:
        value = randint(1,3000)
        if value not in testNumbers:
            testNumbers.append(value)
            count += 1

    #Creates list of train numbers
    trainNumbers = list(range(1,3001))
    for num in testNumbers:
        trainNumbers.remove(num)

    file = open(train_path, "w")
    #Puts test data in each file within the test folder
    for item in testNumbers:
        data = lc.getline('fulldataLabeled.txt', item)

        file.write(data)
    file.close()

    file = open(test_path, "w")
    #Puts train data in each file within the train folder
    for item in trainNumbers:
        data = lc.getline('fulldataLabeled.txt', item)
        
        file.write(data)
    file.close()

#T3
def create30trainingSets():
    #TODO: make this not run if file already exists
    parent_directory = "trainingSets"
    if not os.path.isdir(parent_directory):
        os.mkdir(parent_directory)

    subfolders = ["size2600TrainingSets", "size1300TrainingSets", "size650TrainingSets"]
    for item in subfolders:
        subfolder_path = os.path.join(parent_directory, item)
        os.mkdir(subfolder_path)

    sizes = [2600, 1300, 650]
    for size in sizes:
        for version_num in range(1,11):
            path1 = os.path.join(parent_directory, "size" + str(size) + "TrainingSets")
            path2 = os.path.join(path1, "train" + str(version_num) + ".txt")
            file = open(path2, "w")
            for line_num in range(1,size+1):
                value = randint(1,2600)
                data = lc.getline('trainMaster.txt', value)

                file.write(data)
            file.close()

if __name__ == "__main__":
    # imdb_path = sys.argv[1]
    # train_path = sys.argv[2]
    train_path = "/home/hpc/giordm10/CSC427/NLP-Project-4/trainMaster.txt"
    # test_path = sys.argv[3]
    test_path = "/home/hpc/giordm10/CSC427/NLP-Project-4/testMaster.txt"

    creatOneFile()
    normalize()
    trainTestSplit()
    create30trainingSets()




#removes all old files from the desired train and test folder
def reset():
    dir = train_path
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = test_path
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

# generates the appropriate files based on the imdb62.txt
def main_old():
    testNumbers = []
    count = 0
    #Creates list of test numbers
    while count < 100:
        value = randint(1,1000)
        if value not in testNumbers:
            testNumbers.append(value)
            count += 1

    #Creates list of train numbers
    trainNumbers = list(range(1,1001))
    for num in testNumbers:
        trainNumbers.remove(num)

    #Puts test data in each file within the test folder
    for author in range(0,62):
        for item in testNumbers:
            line_num = item + (author * 1000)
            data = lc.getline('imdb62.txt', line_num)

            split_data = data.split("\t")

            file_path = test_path + split_data[1]

            file = open(file_path + ".txt", "a")
            file.write(split_data[5])
            file.close()

    #Puts train data in each file within the train folder
    for author in range(0,62):
        for item in trainNumbers:
            line_num = item + (author * 1000)
            data = lc.getline('imdb62.txt', line_num)

            split_data = data.split("\t")

            file_path = train_path + split_data[1]

            file = open(file_path + ".txt", "a")
            file.write(split_data[5])
            file.close()
    