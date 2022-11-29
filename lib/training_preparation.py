import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def limit_by_type(df, limit, sort_cname, type_cname):
    df_concat = df.copy()
    df_concat = df_concat.sort_values(by=[sort_cname], ascending=False)
    df_new = pd.DataFrame(columns = df_concat.columns.tolist())
    set_of_document_types = set(df_concat[type_cname].tolist())
    for dtype in set_of_document_types:
        df_new = pd.concat([df_new, df_concat[df_concat[type_cname] == dtype].iloc[:limit]])
    return df_new


def count_distinct_words(filtered_words):
    features_freq = defaultdict(int)
    for w in filtered_words.split():
        features_freq[w] += 1
    return dict(features_freq)


def oversample_by_type(df, document_type_column_name='id_typ_dokument'):
    df_count = df.groupby([document_type_column_name]).size()
    max_count_of_documents = df_count.max(axis=0)

    for id_typ_document, count_of_documents_by_type in df_count.iteritems():
        if count_of_documents_by_type != max_count_of_documents:
            random_sample_to_append = max_count_of_documents - count_of_documents_by_type
            current_df = df[(df[document_type_column_name] == id_typ_document)]
            random_df = current_df.sample(n=random_sample_to_append, replace=True)
            df = df.append(random_df, ignore_index=True)
    return df


def oversample_by_db(df, document_type_column_name='id_typ_dokument', db_type_column_name='db_type'):
    df_count = df.groupby([document_type_column_name, db_type_column_name]).size().unstack(fill_value=0)
    max_count_of_documents = max(df_count.max(axis=0).tolist())

    for id_typ_document, count_of_documents_by_db in df_count.iterrows():
        for column_name in df_count.columns:
            count_of_documents = count_of_documents_by_db[column_name]
            random_sample_to_append = max_count_of_documents - count_of_documents if count_of_documents != 0 else 0
            current_df = df[(df[document_type_column_name] == id_typ_document) & (df[db_type_column_name] == column_name)]
            random_df = current_df.sample(n=random_sample_to_append, replace=True)
            df = df.append(random_df, ignore_index=True)

    return oversample_by_type(df, document_type_column_name)


def get_tt_stratified_split(y, splits=5, random_state=1):
    y = y.astype(int)
    df = pd.DataFrame()
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    for idx, slices in enumerate(skf.split(np.zeros(len(y)), y)):
        df[f'train_{idx}'] = pd.Series(slices[0])
        df[f'test_{idx}'] = pd.Series(slices[1])
    return df