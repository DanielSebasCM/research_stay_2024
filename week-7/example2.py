import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Get word vector for a specific word
word_vector = model['cat']

test_words = ['gato', 'perro', 'casa', 'avión']

# Find most similar words to 'cat'
for word in test_words:
    similar_words = model.most_similar(word, topn=5)

    print("\nMost Similar Words to " + word + ":") 
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.2f}")

