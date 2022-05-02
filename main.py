def make_vocab():
    all_words = []
    vocabulary = []
    with open("fulldataLabeled.txt") as f:
        for review in f:
            review = review.split()
            for word in review:
                #adds every word of the reviews to one array
                all_words.append(word)
                if word not in vocabulary:
                    #gets every unique word in the reviews
                    vocabulary.append(word)

    return vocabulary

def calculate_priors():
    all_ratings = []
    positive_count = 0
    negative_count = 0
    positive_prob = 0
    negative_prob = 0

    # counting the number of positive and negative reviews
    with open("fulldataLabeled.txt") as f:
        for review in f:
            rating = review.split()[-1]
            if rating == "0":
                negative_count += 1
            else:
                positive_count += 1
            all_ratings.append(rating)
    
    positive_prob = positive_count / len(all_ratings)
    negative_prob = negative_count / len(all_ratings)

    return positive_prob, negative_prob

if __name__ == "__main__":
    vocabulary = make_vocab()
    
    positive_prob, negative_prob = calculate_priors()
