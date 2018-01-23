import statistics
import pandas as pd
from string import punctuation
from collections import Counter
from operator import itemgetter
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emot


def sentence_based_features(doc):
    sents = list(doc.sents)
    nr_sents = len(sents)
    nr_words_l = [len(s) for s in doc.sents]
    avg_nr_word_per_sent = sum(nr_words_l) / nr_sents
    if len(nr_words_l) > 1:
        std_nr_word_per_sent = statistics.stdev(nr_words_l)
    else:
        std_nr_word_per_sent = 0  # with one sentence email, there is no STD, I give it a "0"

    return pd.Series({"nr_sents": nr_sents,
                      "avg_nr_word_per_sent": avg_nr_word_per_sent,
                      "std_nr_word_per_sent": std_nr_word_per_sent})


def character_based_features(doc):
    nr_chars = 0
    nr_letters = 0
    nr_upper = 0
    nr_lower = 0
    nr_nums = 0
    nr_spaces = 0
    nr_punctuation = 0
    for token in doc:
        for char in str(token):
            nr_chars += 1
            if char.isalpha():
                nr_letters += 1
            if char.isupper():
                nr_upper += 1
            if char.islower():
                nr_lower += 1
            if char.isnumeric():
                nr_nums += 1
            if char.isspace():
                nr_spaces += 1
            if char in punctuation:
                nr_punctuation += 1
    return pd.Series({"nr_chars": nr_chars, "nr_letters": nr_letters, "nr_upper": nr_upper,
                        "nr_lower": nr_lower,  "nr_nums": nr_nums, "nr_spaces": nr_spaces,
                        "nr_punctuation": nr_punctuation})


def punctuationr_based_features(doc):
    nr_commas = 0
    nr_dots = 0
    nr_exclamation = 0
    nr_question = 0
    nr_colons = 0
    nr_semicolons = 0
    nr_hyphens = 0
    for token in doc:
        if str(token) == ",":
            nr_commas += 1
        if str(token) == ".":
            nr_dots += 1
        if str(token) == "!":
            nr_exclamation += 1
        if str(token) == "?":
            nr_question += 1
        if str(token) == ":":
            nr_colons += 1
        if str(token) == ";":
            nr_semicolons += 1
        if str(token) == "-":
            nr_hyphens += 1

    return pd.Series({"nr_commas": nr_commas, "nr_dots": nr_dots, "nr_exclamation": nr_exclamation,
                    "nr_question": nr_question,  "nr_colons": nr_colons, "nr_semicolons": nr_semicolons,
                     "nr_hyphens": nr_hyphens})


def word_based_features(doc):
    long_word = 5
    nr_words = 0
    sum_characters = 0
    avg_characters_per_word = 0
    nr_longwords = 0
    nr_stopwords = 0
    nr_error = 0
    TTR = 0
    hapaxes = []
    HTR = 0
    token_list = []
    sorted_token_frequency = []
    most_frequency = 0
    for token in doc:
        if token.is_alpha:
            nr_words += 1
            sum_characters += len(str(token))
            if len(token) > long_word:
                nr_longwords += 1
            if token.is_stop:
                nr_stopwords += 1
            if not token.vocab:
                nr_error += 1
            token_list.append(str(token))
    if nr_words > 0:
        avg_characters_per_word = sum_characters / nr_words
    else:
        avg_characters_per_word = 0
    if len(token_list) > 0:
        TTR = len(set(token_list))/len(token_list)
        hapaxes = list(filter(lambda x: token_list.count(x) == 1, token_list))
        HTR = len(hapaxes)/len(token_list)
        sorted_token_frequency = sorted(Counter(token_list).items(), key=itemgetter(1), reverse=True)
        most_frequency = sorted_token_frequency[0][1]
    else:
        TTR = 0
        HTR = 0
        most_frequency = 0
    return pd.Series({"nr_words": nr_words, "avg_characters_per_word": avg_characters_per_word,
                    "nr_longwords": nr_longwords,  "nr_stopwords": nr_stopwords, "nr_error": nr_error,
                     "TTR": TTR, "HTR": HTR, "most_frequency": most_frequency})


def paragraph_based_features(content):
    sum_sent = 0
    av_sent = 0
    sum_word = 0
    av_word = 0
    paragraphs = content.split('\n\n')
    paragraphs = [paragraph for paragraph in paragraphs if not (paragraph.isspace()or paragraph == "")]
    for paragraph in paragraphs:
        sentences = re.split('[?!.\n]', paragraph)
        sentences = [sentence for sentence in sentences if not (sentence.isspace()or sentence == "")]
        sum_sent += len(sentences)
        for sentence in sentences:
            words = sentence.split(" ")
            words = [word for word in words if not (word.isspace()or word == "")]
            sum_word += len(words)
    av_sent = sum_sent / len(paragraphs)
    av_word = sum_word / len(paragraphs)
    return pd.Series({"n_paragraphs": len(paragraphs), "av_sent": av_sent, "av_word": av_word})


def syntactic_features(doc):
    pos_list = []
    function_pos_list = ["PRON", "DET", "ADP", "CONJ", "AUX", "INTJ", "PART", "CCONJ", "PART"]
    nr_function = 0
    sum_length_np = 0
    avg_length_np = 0
    np_list = []
    for token in doc:
        pos_list.append(token.pos_)
    for pos in pos_list:
        if pos in function_pos_list:
            nr_function += 1
    for np in doc.noun_chunks:
        sum_length_np += len(np.text)
        np_list.append(np.text)
    if len(np_list) > 0:
        avg_length_np = sum_length_np/len(np_list)
    else:
        avg_length_np = 0
    return pd.Series({"nr_pos": len(set(pos_list)), "nr_function": nr_function, "avg_length_np": avg_length_np})


def semantic_features_content(content):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(content)
    nr_emoticons = 0
    nr_emoticons += len(emot.emoji(content))+len(emot.emoticons(content))
    return pd.Series({"score_semantic": scores['compound'], "nr_emoticons": nr_emoticons})


def semantic_features(doc):
    sid = SentimentIntensityAnalyzer()
    nr_positive_word = 0
    nr_neg_word = 0
    nr_named_entity = 0
    for token in doc:
        if (sid.polarity_scores(str(token))['compound']) >= 0.5:
            nr_positive_word += 1
        elif (sid.polarity_scores(str(token))['compound']) <= -0.5:
            nr_neg_word += 1
        if token.ent_type_ != "":
            nr_named_entity += 1
    return pd.Series({"nr_positive_word": nr_positive_word, "nr_neg_word": nr_neg_word, 
                    "nr_named_entity": nr_named_entity})