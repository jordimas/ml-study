from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# See similar approcach https://medium.com/analytics-vidhya/content-based-recommender-systems-in-python-2b330e01eb80
def main():
    print("Show concepts involved in the recommender")
  
    corpus = [
        'this is one friend',
        'this is two friend',
        #'Everybody loves me.',
        'Everybody loves you.', 
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    print("** TfidfVectorizer")
    print(f"Shape {tfidf.shape}")
    print(tfidf)
    # output (doc, word) = tfif source

    similarity_matrix = linear_kernel(tfidf, tfidf)
    print("** Similarity_matrix")
    print(f"Similarity shape {similarity_matrix.shape}")
    print(similarity_matrix)
 
    print("Similarity [3]")
    print(similarity_matrix[3])

    # Show recommendations
    recommendations_for_doc = 3
    print(f"doc {recommendations_for_doc}: {corpus[recommendations_for_doc]}")
    print("rec for doc")
    print(similarity_matrix[recommendations_for_doc])
    similarity_score = list(enumerate(similarity_matrix[recommendations_for_doc]))
    print("**List similarity_score")
    print(f"{similarity_score}")
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:3]
    print("** Recommendations")
    print(similarity_score)

    

if __name__ == "__main__":
    main()