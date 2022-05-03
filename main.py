from collections import defaultdict

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

def make_vocab():
    with open("testing.txt") as f:
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
    with open("testing.txt") as f:
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

def test_sentences(prior_pos, prior_neg, likelihood_pos, likelihood_neg):
    test_sentence = "predictable with no fun"
    test_array = test_sentence.split()

    final_pos = 1
    final_neg = 1

    for word in test_array:
        if likelihood_neg[word] or likelihood_pos[word] > 0:
            final_pos *= likelihood_pos[word]
            final_neg *= likelihood_neg[word]

    final_pos *= prior_pos
    final_neg *= prior_neg

    print("pos: ", final_pos)
    print("neg: ", final_neg)

if __name__ == "__main__":
    all_words = make_vocab()
    
    positive_prob, negative_prob = calculate_priors()

    positive_likelihoods, negative_likelihoods = calculate_conditional_likelihoods()

    positive_bin_likelihoods, negative_bin_likelihoods = NBBinary()

    for x in range(0,30):
        calculate_priors()
        NBBinary()
    for x in range(0,30):
        calculate_priors()
        calculate_conditional_likelihoods()


    for x in range(0,2):
        if x == 0:
            print("count")
            test_sentences(positive_prob, negative_prob, positive_likelihoods, negative_likelihoods)
        if x == 1:
            print("binary")
            test_sentences(positive_prob, negative_prob, positive_bin_likelihoods, negative_bin_likelihoods)