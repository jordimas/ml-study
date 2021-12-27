from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


def main():
    print("Show concepts involved in the recommender")
    logging.basicConfig(filename="recommender.log", level=logging.DEBUG, filemode="w")

    corpus = [
        'this is one friend',
        'this is two friend',
        'Everybody loves me.',
        'Everybody loves you.'
        #'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    corpus = vectorizer.fit_transform(corpus)
    print("TfidfVectorizer")
    print(corpus.shape)
    print(corpus)
    # output (doc, word) = tfif source

    similary = linear_kernel(corpus, corpus)
    print(similary.shape)
    print(similary)
 
    print("one element")
    print(similary[3])


if __name__ == "__main__":
    main()