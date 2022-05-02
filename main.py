from pkg_resources import NullProvider
from collections import defaultdict

# Global Variables
all_words = defaultdict(lambda: 0)
all_words_negative = defaultdict(lambda: 0)
all_words_positive = defaultdict(lambda: 0)
binarized_negative = defaultdict(lambda: 0)
binarized_positive = defaultdict(lambda: 0)
total_word_count_positive = 0
total_word_count_negative = 0

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
                for word in review_tokenized:
                    all_words_negative[word] += 1
            else:
                positive_count += 1
                total_word_count_positive += (len(review_tokenized))
                for word in review_tokenized:
                    all_words_positive[word] += 1

            all_ratings.append(rating)
    

    positive_prob = positive_count / len(all_ratings)
    negative_prob = negative_count / len(all_ratings)

    return positive_prob, negative_prob

def calculate_conditional_likelihoods(all_words):
    positive_likelihoods = dict.fromkeys(all_words, 0)
    negative_likelihoods = dict.fromkeys(all_words, 0)

    for word in all_words:
        positive_likelihoods[word] = (all_words_positive[word] + 1) / (total_word_count_positive + len(all_words))
        negative_likelihoods[word] = (all_words_negative[word] + 1) / (total_word_count_negative + len(all_words))    

    print(positive_likelihoods)
    print(negative_likelihoods)

def makeBinary(review):
    # make review a dictionary, so it's only unique words
    
    pass

if __name__ == "__main__":
    all_words = make_vocab()
    
    positive_prob, negative_prob = calculate_priors()

    calculate_conditional_likelihoods(all_words)