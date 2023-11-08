import nltk
from nltk import bigrams
from nltk.corpus import reuters
from nltk.probability import FreqDist

# Load and tokenize the corpus
corpus = reuters.sents()
tokenized_corpus = [word.lower() for sentence in corpus for word in sentence]

# Create bigrams from the tokenized corpus
bigram = list(bigrams(tokenized_corpus))

# Calculate the frequency distribution of bigrams
bigram_freq = FreqDist(bigram)

# Context for prediction (the last word of the context is the word to predict)
context = ("the",)

# Predict the next word
next_word = max((bigram for bigram in bigram_freq if bigram[0] == context), key=lambda x: bigram_freq[x])

print(f"Given the context {context}, the predicted next word is: {next_word[1]}")
