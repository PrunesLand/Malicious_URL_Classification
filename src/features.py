import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

def extract_url_features(df):

    df['url_len'] = df['url'].apply(len)
    df['dot_count'] = df['url'].apply(lambda x: x.count('.'))
    df['digit_count'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))

    X = df[['url', 'url_len', 'dot_count', 'digit_count']]
    y = df['type']
    return X, y

def get_target_encoder():
    return LabelEncoder()

def get_woe_encoder():
    return ce.TargetEncoder(cols=['url'])
