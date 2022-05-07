from collections import defaultdict
from gettext import translation
from random import randint
from pathlib import Path
import linecache as lc
import os
import shutil
import string
import matplotlib.pyplot as plt
import sys


# stores all occurrences of every word in the vocabulary
all_words = defaultdict(lambda: 0)
# stores negative and positive occurrences, respectively
all_words_negative = defaultdict(lambda: 0)
all_words_positive = defaultdict(lambda: 0)

# store versions of each review with duplicate words removed
clipped_negative_reviews = defaultdict(lambda: 0)
clipped_positive_reviews = defaultdict(lambda: 0)

total_word_count_positive = 0
total_word_count_negative = 0

binary_negative_counts = defaultdict(lambda: 0)
binary_positive_counts = defaultdict(lambda: 0)

# nested dictionary that, for every line of a bootstrap file, stores the line number where it originally appeared in testMaster.txt
file_linenum_dict = defaultdict(lambda: 0)

# stores delta-f1 measures for how a pair of systems performed on testMaster
delta_x_dict = defaultdict(lambda: 0)
# stores delta-f1 measures for how a pair of systems performed on a bootstrap file
delta_xi_dict = defaultdict(lambda: 0)

f1_pvalue_pairs = defaultdict(lambda: 0)

# for every system, stores pairs of system classification and ground truth (from testMaster.txt)
alltestruns = defaultdict(lambda: 0)

fulldataLabeled_path = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]

class NBModel():
    """
    Model object that represents a NB learned model
    positive_prior - value of a document being positive
    negative_prior - value of a document being negative
    positive_likelihoods - the likelihoods of each word being in a positive class
    negative_likelihoods - the likelihoods of each word being in a positive class
    """
    def __init__(self, positive_prior, negative_prior, positive_likelihoods, negative_likelihoods):
        self.positive_prior = positive_prior
        self.negative_prior = negative_prior
        self.positive_likelihoods = positive_likelihoods
        self.negative_likelihoods = negative_likelihoods

class PairModel():
    """
    model_one - the first model in the pair
    model_two - the second model in the pair
    model_one_number - an identifying number for the first model in the pair
    model_two_number - an identifying number for the second model in the pair
    """
    def __init__(self, model_one, model_two, model_one_number, model_two_number):
        self.model_one = model_one
        self.model_two = model_two
        self.model_one_number = model_one_number
        self.model_two_number = model_two_number

# T2 part one
def createOneFile():
    """
    Takes all of the input files from sentiment labelled sentences folder and creates one 
    long concatentated file used in the rest of the program
    """
    #  List of all files we are tryng to get data from
    original_files = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
    file_paths = []

    dir = "./sentiment labelled sentences"

    # Creates a list of all full path files
    for file in original_files:
        file_paths.append(os.path.join(dir, file))

    # Open fulldataLabeled.txt in write mode
    with open(fulldataLabeled_path, 'w') as outfile:

        # Iterate through list of all file names
        for names in file_paths:

            # Open each file in read mode
            with open(names) as infile:
                
                # take the information from input files and compile them all together into one file
                outfile.write(infile.read())

def normalize():
    """
    normalizes our input data
    - makes all text lowercase
    - removes all punctuation
    """
    text_doc = []
    #normalize texts
    with open(fulldataLabeled_path) as f:
        for line in f.readlines():
                # removes punctuation
                line = line.translate(str.maketrans('', '', string.punctuation))
                # makes everything lowercase
                line = line.lower()
                sentence = line.split()

                text_doc.append(sentence)

    #write information back into a file
    with open(fulldataLabeled_path, "w") as filehandle:
        for review in text_doc:
            for word in review:
                filehandle.write("%s " % word)
            filehandle.write("\n")

#T2 part two
def trainTestSplit():
    """
    Creates the train and test sets
    Creates these sets by taking a random 2600 lines of text from fulldataLabelled.txt and putting them in the train set
    The remaining lines are put into the test set
    trainMaster.txt is where we store train information
    testMaster.txt is where we store test information
    """
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
        data = lc.getline(fulldataLabeled_path, item)

        file.write(data)
    file.close()

    file = open(test_path, "w")
    #Puts train data in each file within the train folder
    for item in trainNumbers:
        data = lc.getline(fulldataLabeled_path, item)
        
        file.write(data)
    file.close()

#T3
def create30trainingSets():
    """
    Creates the 30 training sets from trainMaster.txt
    Creates 10 training sets for each size of 2600, 1300, and 650 and places them in their respective subdirectory
    The files in this folder have a number of lines equivalent to their size
    """
    parent_directory = "trainingSets"
    if not os.path.isdir(parent_directory):
        os.mkdir(parent_directory)
    # check if subfolders exist
    subfolders = ["size2600TrainingSets", "size1300TrainingSets", "size650TrainingSets"]
    for item in subfolders:
        subfolder_path = os.path.join(parent_directory, item)
        if not os.path.isdir(subfolder_path):
            os.mkdir(subfolder_path)

    sizes = [2600, 1300, 650]
    for size in sizes:
        for version_num in range(1,11):
            path1 = os.path.join(parent_directory, "size" + str(size) + "TrainingSets")
            path2 = os.path.join(path1, "train" + str(version_num) + ".txt")
            file = open(path2, "w")
            for line_num in range(1,size+1):
                value = randint(1,2600)
                data = lc.getline(train_path, value)

                file.write(data)
            file.close()

def make_vocab():
    """
    Creates a dictionary that contains every list in the text and its count
    Does not include the last character which is the ground truth value of the review
    Returns:
        A list of every word in the text and its count
    """
    with open(fulldataLabeled_path) as f:
        for review in f:
            review = review.split()[:-1]
            for word in review:
                # dictionary stores the count of occurrences of each word
                all_words[word] += 1

    return all_words

def calculate_priors():
    """
    Calculates the prior probabilities used in calculations
    Return:
        The positive and negative prior probabilities
    """
    all_ratings = []
    positive_count = 0
    negative_count = 0
    positive_prob = 0
    negative_prob = 0

    global total_word_count_positive
    global total_word_count_negative

    # counting the number of positive and negative reviews
    with open("fulldataLabeled.txt") as f:
        for review in f:
            review_tokenized = review.split()
            rating = review_tokenized[-1]
            review_tokenized = review_tokenized[:-1]

            if rating == "0":
                negative_count += 1
                total_word_count_negative += (len(review_tokenized))

                clipped_negative_reviews[review] = dict.fromkeys(review_tokenized)

                for word in review_tokenized:
                    all_words_negative[word] += 1
            else:
                positive_count += 1
                total_word_count_positive += (len(review_tokenized))

                clipped_positive_reviews[review] = dict.fromkeys(review_tokenized)

                for word in review_tokenized:
                    all_words_positive[word] += 1

            all_ratings.append(rating)
    
    # calculates the probability of the word being positive and negative
    positive_prob = positive_count / len(all_ratings)
    negative_prob = negative_count / len(all_ratings)

    return positive_prob, negative_prob

def calculate_conditional_likelihoods():
    """
    Calculates the conditional likelihoods of a word being in either a positive or negative class
    Returns:
        two dictionaries, one for every word and its probability of it being in a positive class and the same for negative (for count)
    """
    positive_likelihoods = defaultdict(lambda: 0)
    negative_likelihoods = defaultdict(lambda: 0)

    for word in all_words:
        positive_likelihoods[word] = (all_words_positive[word] + 1) / (total_word_count_positive + len(all_words))
        negative_likelihoods[word] = (all_words_negative[word] + 1) / (total_word_count_negative + len(all_words))    

    return positive_likelihoods, negative_likelihoods

def NBBinary():
    """
    Calculates NB classifier, but for binary classification
    Returns:
        two dictionaries, one for every word and its probability of being in a positive class and the same for negative (for binary)
    """
    positive_bin_likelihoods = defaultdict(lambda: 0)
    negative_bin_likelihoods = defaultdict(lambda: 0)
    binary_total_count_positive = 0
    binary_total_count_negative = 0
    
    for review in clipped_negative_reviews:
        for word in clipped_negative_reviews[review]:
            binary_negative_counts[word] += 1
            binary_total_count_positive += 1

    for review in clipped_positive_reviews:
        for word in clipped_positive_reviews[review]:
            binary_positive_counts[word] += 1
            binary_total_count_negative += 1

    for word in all_words:
        positive_bin_likelihoods[word] = (binary_positive_counts[word] + 1) / (binary_total_count_positive + len(all_words))
        negative_bin_likelihoods[word] = (binary_negative_counts[word] + 1) / (binary_total_count_negative + len(all_words))   

    return positive_bin_likelihoods, negative_bin_likelihoods 

def test_sentences(prior_pos, prior_neg, likelihood_pos, likelihood_neg):
    """
    Takes the given test file and classifies it using the given NB classifier
    Args:
        prior_pos : the value of the positive prior probability
        prior_neg : the value of the negative prior probability
        likelihood_pos : a list of all the likelihoods of a word being in a positive document
        likelihood_neg : a list of all the likelihoods of a word being in a negative document

    Returns:
        dictionary of all the classifications
    """
    file = open(test_path,"r")
    test_bin_classifications = []
    for line in file:
        test_sentence = line

        test_array = test_sentence.split()

        final_pos = 1
        final_neg = 1

        for word in test_array:
            if likelihood_neg[word] or likelihood_pos[word] > 0:
                final_pos *= likelihood_pos[word]
                final_neg *= likelihood_neg[word]

        final_pos *= prior_pos
        final_neg *= prior_neg

        return_tuple = tuple((1 if final_pos>final_neg else 0,int(test_array[-1])))
        test_bin_classifications.append(return_tuple)
    return test_bin_classifications
    
def create_pairs(allmodels):
    """
    Creates pair objects of NB Classifier objects
    Args:
        allmodels : a list of all 60 models we have created

    Returns:
        a list of all 1770 pairs of models
    """
    #Creates the list of pairs for T6
    alreadyTested = []
    pairs_list = []
    for model in allmodels:
        alreadyTested.append(model)
        for model2 in allmodels:
            if model2 not in alreadyTested:
                
                new_pair = PairModel(allmodels[model], allmodels[model2], model, model2)
                pairs_list.append(new_pair)

    return pairs_list

def calc_f_measure(onetestrun):
    """
    takes in a list of tuples
    List will be 400 tuples, each with a given system output and ground truth
    calculates the f1 measure based on these pairs
    Args:
        onetestrun : a list of 400 tuples

    Returns:
        the f1 measure of a given system
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    f_measure = 0

    for systemOutput,groundTruth in onetestrun:
        if systemOutput == 1 and groundTruth == 1:
            true_positives += 1
        elif systemOutput == 1 and groundTruth == 0:
            false_positives += 1
        elif systemOutput == 0 and groundTruth == 1:
            false_negatives += 1

    recall = true_positives/(true_positives+false_negatives)
    precision = true_positives/(true_positives+false_positives)

    f_measure = (2 * precision * recall) / (precision + recall)

    return f_measure
    
def bootstrap(pairs_list):
    """
    Creates the list of 1000 bootstrap samples for bootstrap sampling
    Calculates the delta f1 measure of two systems in pair, as well as the p-value
    Creates a tuple of these two values used in outputs and graphing
    Args:
        list of all 1770 pairs of NB classifiers

    Returns:
        path to where temporary bootstrap files are located
    """
    # b can be any positive integer but we chose 1000 for our b value
    b = 1000

    directory = "bootstrap_samples"
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    for each in range(1,b+1):
        filename = "bootstrap"+str(each)+".txt"
        fullName = os.path.join(path, filename)
        file = open(fullName,"w+")
        file_linenum_dict[filename] = defaultdict(lambda: 0)
        index = 0
        for number in range(400):
            r1 = randint(1,400)
            data = lc.getline(test_path,r1)
            file.write(data)
            file_linenum_dict[filename][index] = r1
            index += 1
        file.close()

    d3_output_file = open("d3_output.txt", "w")

    i = 1
    count = 0
    for pair in pairs_list:
        print(count)
        print()
        count+=1
        model1 = pair.model_one
        model2 = pair.model_two
        sum = 0.0

        f_measure_a = calc_f_measure(model1)
        f_measure_b = calc_f_measure(model2)
        
        swapped = False
        if f_measure_a < f_measure_b:
            f_measure_a, f_measure_b = f_measure_b, f_measure_a
            swapped = True

        delta_x = f_measure_a - f_measure_b

        delta_x_dict[pair] = delta_x
        
        for filename in os.listdir(path):
            

            filepath = os.path.join(path,filename)
            file = open(filepath,"r")

            a_list = []
            b_list = []

            index = 0
            for line in file:

                if swapped == False:
                    a_list.append(alltestruns[pair.model_one_number][file_linenum_dict[filename][index] - 1])
                    b_list.append(alltestruns[pair.model_two_number][file_linenum_dict[filename][index] - 1])
                else:
                    a_list.append(alltestruns[pair.model_two_number][file_linenum_dict[filename][index] - 1])
                    b_list.append(alltestruns[pair.model_one_number][file_linenum_dict[filename][index] - 1])

                index += 1

            f_measure_axi = calc_f_measure(a_list)
            f_measure_bxi = calc_f_measure(b_list)
        
            delta_xi_dict[filename] = f_measure_axi - f_measure_bxi

            sum += getPValNumerator(delta_xi_dict[filename],delta_x_dict[pair])

        file.close()

        f1_pvalue_pairs[i] = tuple((delta_x_dict[pair],sum/b))
        print(f1_pvalue_pairs[i])

        d3_output_file.write(str(f1_pvalue_pairs[i]) + "\n")
        i += 1

    return path

def getPValNumerator(delta_xi,delta_x):
    return 1 if delta_xi >= 2*delta_x else 0


def graph(pairs_list):
    """
    Creates graphs based on p-value and delta f1 measure
    Args:
        list of all 1770 pairs
    """
    x = []
    y = []

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    index = 0
    for data_point in f1_pvalue_pairs:
        x.append(f1_pvalue_pairs[data_point][0])
        y.append(1 - f1_pvalue_pairs[data_point][1])

        if pairs_list[index].model_one_number <= 29 and pairs_list[index].model_two_number <= 29:
            x1.append(f1_pvalue_pairs[data_point][0])
            y1.append(1 - f1_pvalue_pairs[data_point][1])
        elif pairs_list[index].model_one_number > 29 and pairs_list[index].model_two_number > 29:
            x1.append(f1_pvalue_pairs[data_point][0])
            y1.append(1 - f1_pvalue_pairs[data_point][1])
        else:
            x2.append(f1_pvalue_pairs[data_point][0])
            y2.append(1 - f1_pvalue_pairs[data_point][1])

        index += 1
    plt.scatter(x, y, marker='x', c='black')
    
    plt.xlabel('deltaF1')
    plt.ylabel('1 - pValue')
    plt.title('Plot 1')


    plt.show()

    plt.savefig("plot1.png")

    plt.close()

    plt.scatter(x1, y1, c='blue', marker='x', label='-> same data representations')
    plt.scatter(x2, y2, c='red', marker='o', label='-> different data representations')
    plt.legend()

    # plot
    plt.xlabel("deltaF1")
    plt.ylabel("1 - pValue")
    plt.title("Plot 2")
    plt.show()

    plt.savefig("plot2.png")

    plt.close()

if __name__ == "__main__":

    print(fulldataLabeled_path)
    print(train_path)
    print(test_path)

    createOneFile()
    normalize()
    trainTestSplit()
    create30trainingSets()

    all_words = make_vocab()

    allmodels = defaultdict(lambda: 0)
    i = 0

    trainingSets = "./trainingSets"
    for subfolder in os.listdir(trainingSets):
        subfolder_path = os.path.join(trainingSets, subfolder)

        for trainfile in os.listdir(subfolder_path):
            positive_prob, negative_prob = calculate_priors()
            positive_bin_likelihoods, negative_bin_likelihoods = NBBinary()
            positive_likelihoods, negative_likelihoods = calculate_conditional_likelihoods()

            countModel = NBModel(positive_prob, negative_prob, positive_likelihoods, negative_likelihoods)
            binaryModel = NBModel(positive_prob, negative_prob, positive_bin_likelihoods, negative_bin_likelihoods)

            allmodels[i] = countModel
            allmodels[i+30] = binaryModel

            alltestruns[i] = test_sentences(positive_prob, negative_prob, positive_likelihoods, negative_likelihoods)
            alltestruns[i+30] = test_sentences(positive_prob, negative_prob, positive_bin_likelihoods, negative_bin_likelihoods)
            
            i += 1
    
    pairs_list = create_pairs(alltestruns)

    path_to_delete = bootstrap(pairs_list)

    graph(pairs_list)    
    
    shutil.rmtree(path_to_delete)