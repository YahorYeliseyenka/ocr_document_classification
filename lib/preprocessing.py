import re
from collections import defaultdict


PL_CHARS = b'\xc4\x85\xc4\x87\xc4\x99\xc5\x82\xc5\x84\xc3\xb3\xc5\x9b\xc5\xba\xc5\xbc'.decode('UTF-8')


def lower_text(string):
    return str(string).lower()


def remove_numbers_punctuation_whitespaces(string, min_word_length):
    return re.findall(f"[a-z{PL_CHARS}]{{{min_word_length},}}", string)


def remove_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]


def apply_stempel_stemmer(words, stemmer, min_word_length):
    return [stemmed_word for stemmed_word in [stemmer.stem(word) for word in words] if stemmed_word is not None and len(stemmed_word) > min_word_length]


def apply_spacy_lemmatize(string, lemmatizer):
    return [word.lemma_ for word in lemmatizer(str(string))]


def remove_empty_documents(df, min_count_of_words):
    df['text'] = df['text'].astype('str')
    df['count_of_words'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['count_of_words'] > min_count_of_words]
    df = df[[column_name for column_name in df.columns.tolist() if column_name != 'count_of_words']]
    return df