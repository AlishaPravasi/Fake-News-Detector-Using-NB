# Fake News Detection Project

## Project Overview
This project aims to address the detection of fake news, which has become a significant challenge in the digital media age. With the rapid spread of misinformation, particularly on social media platforms, it's critical to develop automated methods that can help distinguish between credible news sources and fake news.

### Objective
The goal is to build a model that accurately classifies news articles as either real or fake, helping reduce the spread of misinformation.

### Approach
- **Model**: Naive Bayes classifier, well-suited for text classification tasks.
- **Preprocessing**: Tokenization, stop-word removal, and vectorization to prepare text data for model training.
- **Dataset**: Labeled datasets from Kaggle for training and testing.

## Files in this Project
- **`bestModel.model`**: The best Naive Bayes model found for fake news detection.
- **`Smaller_sample.csv`**: Dataset used to evaluate the initial four Naive Bayes models (sourced from Kaggle: [Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)).
- **`WELFake_Dataset.csv`**: Full dataset with 72,134 news articles/headlines used for training and testing (sourced from Kaggle: [Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)).
- **`WELFake_Dataset_Cleaned.csv`**: Cleaned version of the dataset (no special characters, extra spaces, or English stopwords).
- **`main.py`**: Core script to test models, find the best model, train and test it, and output results with graphs.

## How to Run
To execute the fake news detection application, run the following command in your terminal:
```
python3 main.py
```

## Required Libraries
Ensure the following Python libraries are installed (use `pip3 install <library>`):
- `pandas`
- `re`
- `pickle`
- `numpy`
- `matplotlib`
- `sklearn` (scikit-learn)
- `nltk`

## Additional Setup
For NLTK stopwords, add the following to the top of `main.py` if stopwords are not available on your machine. This will prompt the NLTK downloader GUI to fetch "stopwords":

```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```
