from collections import defaultdict
# from collections import OrderedDict
from gettext import translation
from random import randint
from pathlib import Path
import linecache as lc
import os
from re import S
import shutil
import string

# Global Variables
all_words = defaultdict(lambda: 0)
all_words_negative = defaultdict(lambda: 0)
all_words_positive = defaultdict(lambda: 0)

clipped_negative_reviews = defaultdict(lambda: 0)
clipped_positive_reviews = defaultdict(lambda: 0)

total_word_count_positive = 0
total_word_count_negative = 0

binary_negative_counts = defaultdict(lambda: 0)
binary_positive_counts = defaultdict(lambda: 0)

file_linenum_dict = defaultdict(lambda: 0)
models_on_testmaster = defaultdict(lambda: 0)
classified_sentences_count = defaultdict(lambda: 0)
classified_sentences_binary = defaultdict(lambda: 0)
delta_x_dict = defaultdict(lambda: 0)
delta_xi_dict = defaultdict(lambda: 0)

# will all p-values we store be the same?
f1_pvalue_pairs = defaultdict(lambda: 0)

alltestruns = defaultdict(lambda: 0)

class NBModel():
    def __init__(self, positive_prior, negative_prior, positive_likelihoods, negative_likelihoods):
        self.positive_prior = positive_prior
        self.negative_prior = negative_prior
        self.positive_likelihoods = positive_likelihoods
        self.negative_likelihoods = negative_likelihoods

class PairModel():
    def __init__(self, model_one, model_two, model_one_number, model_two_number):
        self.model_one = model_one
        self.model_two = model_two
        self.model_one_number = model_one_number
        self.model_two_number = model_two_number

# T2 part one
def createOneFile():
    #  List of all files we are tryng to get data from
    original_files = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
    file_paths = []

    # TODO: make this user input
    dir = "/home/hpc/lishkom1/Project4/sentiment labelled sentences"

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
                data = lc.getline('trainMaster.txt', value)

                file.write(data)
            file.close()

def make_vocab():
    with open("fulldataLabeled.txt") as f:
        for review in f:
            review = review.split()[:-1]
            for word in review:
                # dictionary stores the count of occurrences of each word
                all_words[word] += 1

    return all_words

def calculate_priors():
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

                #review_tokenized = dict.fromkeys(review_tokenized)
                clipped_negative_reviews[review] = dict.fromkeys(review_tokenized)

                for word in review_tokenized:
                    all_words_negative[word] += 1
            else:
                positive_count += 1
                total_word_count_positive += (len(review_tokenized))

                #review_tokenized = dict.fromkeys(review_tokenized)
                clipped_positive_reviews[review] = dict.fromkeys(review_tokenized)

                for word in review_tokenized:
                    all_words_positive[word] += 1

            all_ratings.append(rating)
    
    # calculates the probability of the word being positive and negative
    positive_prob = positive_count / len(all_ratings)
    negative_prob = negative_count / len(all_ratings)

    return positive_prob, negative_prob

def calculate_conditional_likelihoods():
    positive_likelihoods = defaultdict(lambda: 0)
    negative_likelihoods = defaultdict(lambda: 0)

    for word in all_words:
        positive_likelihoods[word] = (all_words_positive[word] + 1) / (total_word_count_positive + len(all_words))
        negative_likelihoods[word] = (all_words_negative[word] + 1) / (total_word_count_negative + len(all_words))    

    return positive_likelihoods, negative_likelihoods

def NBBinary():
    # make review a dictionary, so it's only unique words
    
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

def test_sentences(prior_pos, prior_neg, likelihood_pos, likelihood_neg, whichModel):
    file = open("testMaster.txt","r")
    test_bin_classifications = []
    for line in file:
        test_sentence = line

        #test_sentence = "predictable with no fun"
        test_array = test_sentence.split()

        # print(test_array)
        # print(test_array[-1])

        final_pos = 1
        final_neg = 1

        for word in test_array:
            if likelihood_neg[word] or likelihood_pos[word] > 0:
                final_pos *= likelihood_pos[word]
                final_neg *= likelihood_neg[word]

        final_pos *= prior_pos
        final_neg *= prior_neg

        return_tuple = tuple((1 if final_pos>final_neg else 0,int(test_array[-1])))
        
        if whichModel < 30:
            classified_sentences_count[line] = defaultdict(lambda: 0)
            classified_sentences_count[line][whichModel] = return_tuple
        else:
            classified_sentences_binary[line] = defaultdict(lambda: 0)
            classified_sentences_binary[line][whichModel] = return_tuple
            
        # print(classified_sentences_count)

        test_bin_classifications.append(return_tuple)

        #print("pos: ", final_pos)
        #print("neg: ", final_neg)
    #print(str(len(test_bin_classifications)))
    #models_on_testmaster[whichModel] = test_bin_classifications
    return test_bin_classifications
    
def create_pairs(allmodels):
    #Creates the list of pairs for T6
    alreadyTested = []
    pairs_list = []
    for model in allmodels:
        # print("model: ", model)
        #print(allmodels[model])
        alreadyTested.append(model)
        # print(model)
        for model2 in allmodels:
            if model2 not in alreadyTested:
                
                new_pair = PairModel(allmodels[model], allmodels[model2], model, model2)
                pairs_list.append(new_pair)
                #print("The output is: " + str(allmodels[model][0]),str(allmodels[model2][0]))

    # print(pairs_list)
    return pairs_list

def calc_f_measure(onetestrun):
    # for item in alltestruns.keys():
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    f_measure = 0
    # print("Onetestrun: is " + str(type(onetestrun[0])))
    for systemOutput,groundTruth in onetestrun:
        # systemOutput = alltestruns[0]
        # groundTruth = alltestruns[1]
        # print (systemOutput,groundTruth)
        if systemOutput == 1 and groundTruth == 1:
            true_positives += 1
        elif systemOutput == 1 and groundTruth == 0:
            false_positives += 1
        elif systemOutput == 0 and groundTruth == 1:
            false_negatives += 1

    recall = true_positives/(true_positives+false_negatives)
    precision = true_positives/(true_positives+false_positives)

    f_measure = (2 * precision * recall) / (precision + recall)

    # print("f_measure: ", f_measure)
    # print("true_p: ", true_positives)
    # print("false_p: ",false_positives)
    # print("false_n: ",false_negatives)
    # print()
    return f_measure
    
def bootstrap(pairs_list):
    b = 1000
    #count = 0
    #sum = 0.0
    #delta_x = 0

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
            #print(r1)
            data = lc.getline('testMaster.txt',r1)
            file.write(data)
            file_linenum_dict[filename][index] = r1
            index += 1
        file.close()

    i = 1
    count = 0
    for pair in pairs_list:
        print(count)
        print()
        count+=1
        model1 = pair.model_one
        model2 = pair.model_two
        sum = 0.0

        # if count == 0:
        #     print("Model 1 is: " + str(model1))
        #     print("Model 2 is: " + str(model2))
        #     count+=1

        f_measure_a = calc_f_measure(model1)
        # print("f_measure_a: ", f_measure_a)
        f_measure_b = calc_f_measure(model2)
        # print("f_measure_b: ", f_measure_b)
        
        delta_x = f_measure_a - f_measure_b

        delta_x_dict[pair] = delta_x

        #print(delta_x)

        #file = open("testMaster.txt","r")

        # dir1 = open(path,"w")
        #bootstrap_dict = defaultdict(lambda: 0)
        
        for filename in os.listdir(path):
            

            filepath = os.path.join(path,filename)
            file = open(filepath,"r")

            a_list = []
            b_list = []

            # for now, this is storing the same thing in different lists, and pulling from them
            # we can't make a_list for count and b_list for binary, because we need to compare
            # two systems in a pair. It's not guaranteed that one is count and the other is binary.
            # a_list should be for model 1. b_list should be for model 2
            # the problem is, they're the same list
            index = 0
            for line in file:
                # print(line)
                # print()

                # index = 0
                # Loop through the file line by line
                # file1 = open("testMaster.txt", "r")
                # for line1 in file1:  
                        
                #     # checking string is present in line or not
                #     if line in line1:
                #         break
                #     index += 1
                #     # print(index)

                #print(type(file_linenum_dict[filename][line]))
                # print("index: ", index)
                # print("file_linenum: ", file_linenum_dict[filename][index])
                # if index == 10:
                #     index -= 1
                
                a_list.append(alltestruns[pair.model_one_number][file_linenum_dict[filename][index] - 1])
                # print(file_linenum_dict[filename][line])
                # print("filename: ", type(filename))
                # print("line: ", type(line))
                # print(alltestruns[pair.model_one_number][file_linenum_dict[filename][line]])
                b_list.append(alltestruns[pair.model_two_number][file_linenum_dict[filename][index] - 1])

                index += 1

            # print(a_list)
            # print()
            # print(b_list)
            f_measure_axi = calc_f_measure(a_list)
            f_measure_bxi = calc_f_measure(b_list)
            # if a_list == b_list:
            #     print("true")
            # else:
            #     print("false")
            # print("f_measure_axi: ", f_measure_axi)
            # print("f_measure_bxi: ", f_measure_bxi)
            # print()
        
            delta_xi_dict[filename] = f_measure_axi - f_measure_bxi
            #print("Deltas are: " + (str(f_measure_a-f_measure_b)))
            
            # The delta calculated for one bootstrap file is compared to the delta for each pair
            # in pairs_list... we add up each sum 1000 times, then divide by b, like in our notes
            sum += getPValNumerator(delta_xi_dict[filename],delta_x_dict[pair])
            # print("sum is: " + str(sum))
        #print("The sum is: " + str(sum))
        file.close()
    # print("sum is: " + str(sum))
    #print("f-measure is: " + str(delta_x_dict[pair])+" "+str(sum/b))
        f1_pvalue_pairs[i] = tuple((abs(delta_x_dict[pair]),sum/b))
        #print("sum: ", sum)
        print(f1_pvalue_pairs[i])
        i += 1

    #return sum/b

def getPValNumerator(delta_xi,delta_x):
    # print("delta_xi: ", delta_xi)
    # print("delta_x: ", delta_x)
    return 1 if delta_xi >= 2*delta_x else 0


if __name__ == "__main__":

    # imdb_path = sys.argv[1]
    # train_path = sys.argv[2]
    train_path = "/home/hpc/lishkom1/Project4/trainMaster.txt"
    # test_path = sys.argv[3]
    test_path = "/home/hpc/lishkom1/Project4/testMaster.txt"

    testfile = "testMaster.txt"
    trainingSets = Path("/home/hpc/lishkom1/Project4/trainingSets")
    # createOneFile()
    # normalize()
    # trainTestSplit()
    # create30trainingSets()

    all_words = make_vocab()

    allmodels = defaultdict(lambda: 0)
    i = 0
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

            # print("count ", trainfile)
            # run this in bootstrap
            alltestruns[i] = test_sentences(positive_prob, negative_prob, positive_likelihoods, negative_likelihoods,i)
            
            # print("binary")
            alltestruns[i+30] = test_sentences(positive_prob, negative_prob, positive_bin_likelihoods, negative_bin_likelihoods,i+30)
            
            i += 1
            # print(i)
            
            # print()
    
    # print(alltestruns)
    # print("Starts here: ")
    # for i in alltestruns:
    #     print(i)
    # od = OrderedDict(sorted(alltestruns.items()))
    pairs_list = create_pairs(alltestruns)
    # print(len(pairs_list))
    # od = OrderedDict(sorted(allmodels.items()))
    # for model in od:
    #     print(od[model].positive_prior)
    # print(pairs_list[0])
    # calc_f_measure(alltestruns)

    bootstrap(pairs_list)
    # print(f1_pvalue_pairs)
    # print(alltestruns[0][0][0])
    # print(pairs_list[0].model_one)
    # print(pairs_list)

    # print("COUNT MODELS")
    # print(nbcountmodels)
    # print()
    # print("BINARY MODELS")
    # print(nbbinarymodels)

    # print(nbcountmodels[0].positive_prior)
    # print(nbcountmodels[0].negative_prior)
    # print(nbcountmodels[0].positive_likelihoods)
    # print(nbcountmodels[0].negative_likelihoods)