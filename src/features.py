"""
This file contains functions to extract features for classification
"""

from string import punctuation
from collections import Counter
from operator import itemgetter
import re
import pandas as pd
import statistics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emot


def sentence_based_features(doc):
    """
    Extract sentence based features, such as:
    - number of sentences in the email
    - avrage number of word per sentence in the email
    - std of number of word per sentence in the email
    """
    sents = list(doc.sents)
    nr_sents = len(sents)
    nr_words_l = [len(s) for s in doc.sents]
    avg_nr_word_per_sent = sum(nr_words_l) / nr_sents
    if len(nr_words_l) > 1:
        std_nr_word_per_sent = statistics.stdev(nr_words_l)
    else:
        # with one sentence email, there is no STD, I give it a "0"
        std_nr_word_per_sent = 0

    return pd.Series({"nr_sents": nr_sents,
                      "avg_nr_word_per_sent": avg_nr_word_per_sent,
                      "std_nr_word_per_sent": std_nr_word_per_sent})


def character_based_features(doc):
    """
    Extract character based features, such as:
    - number of characters per email
    - number of alphbet letters per email
    - number of upper alphbet letter per email
    - number of lower alphbet letter per email
    - number of numerical characters per email
    - number of space per email
    - number of punctuation per email
    """
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

    return pd.Series({"nr_chars": nr_chars,
                      "nr_letters": nr_letters,
                      "nr_upper": nr_upper,
                      "nr_lower": nr_lower,
                      "nr_nums": nr_nums,
                      "nr_spaces": nr_spaces,
                      "nr_punctuation": nr_punctuation})


def punctuation_based_features(doc):
    """
    Extract punctuation based features, such as:
    - number of commas per email
    - number of dots per email
    - number of exclamation marks per email
    - number of question marks per email
    - number of colons per email
    - number of semicolons per email
    - number of hyphens per email
    """
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

    return pd.Series({"nr_commas": nr_commas, 
                      "nr_dots": nr_dots, 
                      "nr_exclamation": nr_exclamation,
                      "nr_question": nr_question,  
                      "nr_colons": nr_colons, 
                      "nr_semicolons": nr_semicolons,
                      "nr_hyphens": nr_hyphens})


def word_based_features(doc):
    """
    Extract word based features, such as:
    - number of words per email
    - avrage number of characters per word in the email
    - number of longwords(more than 5 letters) per email
    - number of stopwords per email
    - number of spelling error per email
    - The TTR (type-token ratio) in the email
    - The HTR (hapax legomena/token ratio) in the email
    - frequency of most-frequent words in the email
    """
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

    return pd.Series({"nr_words": nr_words,
                      "avg_characters_per_word": avg_characters_per_word,
                      "nr_longwords": nr_longwords,
                      "nr_stopwords": nr_stopwords,
                      "nr_error": nr_error,
                      "TTR": TTR,
                      "HTR": HTR,
                      "most_frequency": most_frequency})


def paragraph_based_features(content):
    """
    Extract paragraph based features, such as:
    - number of paragraphs per email
    - avrage number of sentences per paragraph
    - avrage number of words per paragraph
    """
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

    return pd.Series({"n_paragraphs": len(paragraphs), 
                      "av_sent": av_sent, 
                      "av_word": av_word})


def syntactic_features(doc):
    """
    Extract syntactic features, such as:
    - number of diffrent part-of-speech per email
    - number of function words per email
    - average length of noun/verb phrases per email
    """
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

    return pd.Series({"nr_pos": len(set(pos_list)), 
                      "nr_function": nr_function, 
                      "avg_length_np": avg_length_np})


def semantic_features_content(content):
    """
    Extract semantic features, such as:
    - overall sentiment score per email
    - number of emoticons per email
    - number of greeting per email
    """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(content)
    greeting_words_list = ["Dear", "To Whom It May Concern", "Hello", "Hi"]
    nr_greeting = 0
    for word in content.split():
        if word in greeting_words_list:
            nr_greeting += 1
    nr_emoticons = 0
    nr_emoticons += len(emot.emoji(content))+len(emot.emoticons(content))

    return pd.Series({"score_semantic": scores['compound'],
                      "nr_emoticons": nr_emoticons,
                      "nr_greeting": nr_greeting})


def semantic_features(doc):
    """
    Extract semantic features, such as:
    - number of positive words per email
    - number of negative words per email
    - number of named entities per email
    """
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

    return pd.Series({"nr_positive_word": nr_positive_word,
                      "nr_neg_word": nr_neg_word,
                      "nr_named_entity": nr_named_entity})