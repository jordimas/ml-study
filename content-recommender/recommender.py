import json
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def load_programs():
    with open("../../softcatala-web-dataset/dataset/programes.json", 'r') as j:
        programs = json.loads(j.read())

    print(f"Loaded {len(programs)} programs")
    return programs

def main():
    print("Convers a Wordpress export to a JSON usable dataset")
    logging.basicConfig(filename="recommender.log", level=logging.DEBUG, filemode="w")

    # Load all words
    programs = load_programs()

    texts = []
    score_idx = 0
    firefox_score_idx = 0
    score_idx_to_id = {}
    for program in programs:
        score_idx_to_id[score_idx] = program
        program['score_idx'] = score_idx
        texts.append(program['content'])
        if int(program['id']) == 6750:
            print(f"Firefox score_id: {score_idx} - '{program['title']}'")
            firefox_score_idx = score_idx

        score_idx += 1

    vectorizer = TfidfVectorizer()
    corpus = vectorizer.fit_transform(texts)
    print("TfidfVectorizer")
    print(corpus.shape)
    print(corpus)
    # output (doc, word) = tfif source

    similarity_matrix = linear_kernel(corpus, corpus)
    print(similarity_matrix.shape)
    print(similarity_matrix)

    similarity_score = list(enumerate(similarity_matrix[firefox_score_idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:6]
    for score_idx in similarity_score:
        score_idx = score_idx[0]
#        print(f"score_idx: {score_idx}")
        program = score_idx_to_id[score_idx]
        print(program['title'])


    #mapping = pd.Series(articles['id'], index = articles['title'])

    #Recommend ID 

if __name__ == "__main__":
    main()