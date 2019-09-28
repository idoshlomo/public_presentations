"""script that represents any custom code"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.externals import joblib
from scipy.spatial.distance import cosine
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_model(train_data):
    train_text_corpus = [x for x in train_data['book_title'].tolist() if pd.notnull(x)]
    book_title_model = TfidfVectorizer()
    book_title_model = book_title_model.fit(train_text_corpus)
    return book_title_model


def load_model(model_dir):
    mdl = joblib.load(os.path.join(model_dir, "book_title_model.joblib"))
    return mdl


def use_model(input_data, model):
    input_1 = input_data['arg1']
    input_2 = input_data['arg2']
    logger.info('input 1: {}, input 2: {}'.format(input_1, input_2))
    score = 1 - cosine(model.transform([input_1]).todense(),
                       model.transform([input_2]).todense())
    return score


def save_model(mdl, model_dir):
    joblib.dump(mdl, os.path.join(model_dir, "book_title_model.joblib"))


