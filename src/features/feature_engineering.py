import os
import re
import sys
import logging
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Text, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _lemmatization(text: Text) -> Text:
    lemmatizer = WordNetLemmatizer()
    text_list: List[str] = text.split()
    lemmatized_list: List[str] = [lemmatizer.lemmatize(y) for y in text_list]
    return " ".join(lemmatized_list)

def _remove_stop_words(text: Text) -> Text:
    stop_words = set(stopwords.words("english"))
    text_list: List[str] = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text_list)

def _removing_numbers(text: Text) -> Text:
    return ''.join([i for i in text if not i.isdigit()])

def _lower_case(text: Text) -> Text:
    text_list: List[str] = text.split()
    lower_list: List[str] = [y.lower() for y in text_list]
    return " ".join(lower_list)

def _removing_punctuations(text: Text) -> Text:
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _removing_urls(text: Text) -> Text:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df.content = df.content.apply(lambda content: _lower_case(content))
    df.content = df.content.apply(lambda content: _remove_stop_words(content))
    df.content = df.content.apply(lambda content: _removing_numbers(content))
    df.content = df.content.apply(lambda content: _removing_punctuations(content))
    df.content = df.content.apply(lambda content: _removing_urls(content))
    df.content = df.content.apply(lambda content: _lemmatization(content))
    return df

def data_preprocessing() -> None:
    """
    Loads raw data, preprocesses it, and saves it to the preprocessed data directory.
    """
    try:
        logging.info("Downloading NLTK assets")
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)

        logging.info("Loading raw data")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        
        logging.info("Normalizing text data")
        train_processed_data = _normalize_text(train_data)
        test_processed_data = _normalize_text(test_data)
        
        data_path = os.path.join("data", "preprocessed")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logging.info("Successfully saved preprocessed data")

    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}. Please run the data ingestion step.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    data_preprocessing()