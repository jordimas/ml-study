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

def get_array_with_texts(programs):
    texts = []
    for program in programs:
        texts.append(program['content'])

    return texts

def add_score_idx(programs):
    score_idx = 0
    for program in programs:
        program['score_idx'] = score_idx
        if int(program['id']) == 6750:
            print(f"Firefox score_id: {score_idx} - '{program['title']}'")
            firefox_score_idx = score_idx

        score_idx += 1

def get_score_id_for_program_name(programs, program_name):
    for program in programs:
        if program['title'] == program_name:
            return program['score_idx']

    return -1

def get_program_from_score_idx(programs, score_idx):
    for program in programs:
        if program['score_idx'] == score_idx:
            return program

    return None

def show_recommendations_for_program_name(programs, program_name, similarity_matrix):
    score_idx = get_score_id_for_program_name(programs, program_name)
    
    similarity_score = list(enumerate(similarity_matrix[score_idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:6]
    print(f"{program_name}")
    for score in similarity_score:
        score_idx = score[0]
#        print(f"score_idx: {score_idx}")
        program = get_program_from_score_idx(programs, score_idx)
        #print(f" {program['title']} - {score}")
        print(f" {program['title']}")

    print("")

def read_stop_words():
    with open('stopwords-ca.txt') as f:
        lines = [line.rstrip() for line in f]

    return lines

def main():
    print("Convers a Wordpress export to a JSON usable dataset")
    logging.basicConfig(filename="recommender.log", level=logging.DEBUG, filemode="w")

    # Load all words
    programs = load_programs()
    texts = get_array_with_texts(programs)
    add_score_idx(programs)


    stop_words = read_stop_words()
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    corpus = vectorizer.fit_transform(texts)
    print("TfidfVectorizer")
    print(corpus.shape)
    print(corpus)
    # output (doc, word) = tfif source

    similarity_matrix = linear_kernel(corpus, corpus)
    print(similarity_matrix.shape)
    #print(similarity_matrix)

    show_recommendations_for_program_name(programs, "Inkscape", similarity_matrix)

    #for program in programs:
    #    show_recommendations_for_program_name(programs, program['title'], similarity_matrix)        


if __name__ == "__main__":
    main()