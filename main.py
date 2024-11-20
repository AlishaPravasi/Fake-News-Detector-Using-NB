import pandas as pd
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score

# ensure NLTK stopwords are available if not, download nltk('stopwords')
from nltk.corpus import stopwords

path = "Smaller_sample.csv" 
label = "label"
features = ["title", "text"]
classifiers = {
    'Multinomial NB': MultinomialNB(),
    'Complement NB': ComplementNB(),
    'Gaussian NB': GaussianNB(),
    'Bernoulli NB': BernoulliNB()
}

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


def find_best_nb_model(X, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # vectorize the data once, as k-folds will split the indices
    X_vec = vectorizer.fit_transform(X).toarray()  # Convert sparse to dense

    precision_scores = []
    recall_scores = []
    auroc_scores = []
    model_names = []
    best_nb_name = None
    best_nb = None
    highest_avg_accuracy = 0

    # define hyperparameter grids for each classifier
    param_grids = {
        'Multinomial NB': {'alpha': [0.01, 0.1, 1, 10], 'fit_prior': [True, False]},
        'Bernoulli NB': {'alpha': [0.01, 0.1, 1, 10],  'fit_prior': [True, False]},
        'Complement NB': {'alpha': [0.01, 0.1, 1, 10], 'fit_prior': [True, False], 'norm': [True, False]},
        'Gaussian NB': {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}
    }

    for clf_name, clf in classifiers.items():
        if clf_name in param_grids:
            # grid search for the current classifier
            grid_search = GridSearchCV(
                estimator=clf,
                param_grid=param_grids[clf_name],
                cv=kf,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_vec, y)
            
            # retrieve the best model and its parameters
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            best_auroc = 0
            
            # compute cross-validation predictions
            y_pred = cross_val_predict(best_model, X_vec, y, cv=kf)
            y_pred_proba = cross_val_predict(best_model, X_vec, y, cv=kf, method="predict_proba")[:, 1]

            # calculate metrics
            mean_auroc = roc_auc_score(y, y_pred_proba)
            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred)
            accuracy_scores = cross_val_score(best_model, X_vec, y, cv=kf, scoring='accuracy')
            std_accuracy = accuracy_scores.std()

            precision_scores.append(precision)
            recall_scores.append(recall)
            auroc_scores.append(mean_auroc)
            model_names.append(clf_name)
            
            print(f"\n{clf_name}:")
            print(f"Best Parameters: {best_params}")
            print(f"Best Cross-Validated Accuracy: {best_score:.4f}")
            print(f"Mean AUROC: {mean_auroc:.4f}")
            print(f"Precision Score: {precision:.4f}")
            print(f"Recall Score: {recall:.4f}")
            print(f"Accuracy Standard Deviation: {std_accuracy:.4f}")
            print(f"Confusion Matrix:")
            print(conf_matrix)
            print(f"Classification Report:")
            print(classification_report(y, y_pred))
            
            # update the best model if it has the highest average accuracy
            if best_score > highest_avg_accuracy:
                highest_avg_accuracy = best_score
                best_nb_name = clf_name
                best_nb = best_model

    # plot the bar graph
    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_scores, width, label='Precision')
    plt.bar(x, recall_scores, width, label='Recall')
    plt.bar(x + width, auroc_scores, width, label='AUROC')

    plt.xlabel('Sets')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and AUROC by Bernoulli NB Sets')
    plt.xticks(x, model_names)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nHighest accuracy model is {best_nb_name} with an accuracy of {highest_avg_accuracy:.4f}.")
    return best_nb, best_nb_name, best_params

def train_model(X, y, model, model_name, model_params, test_size=0.2):
    # split the data into training and final test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # vectorize the data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)  
    X_test_vec = vectorizer.transform(X_test) 
    
    precision_scores = []
    recall_scores = []
    auroc_scores = []
    set_names = ["Training Set", "Testing Set"]

    # apply the best parameters to the model
    model.set_params(**model_params)
    
    # stratifiedKFold cross-validation for the training set
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # cross-validation predictions
    y_train_pred_proba = cross_val_predict(model, X_train_vec, y_train, cv=kf, method="predict_proba")[:, 1]
    y_train_pred = cross_val_predict(model, X_train_vec, y_train, cv=kf)
    
    # training set metrics
    train_mean_auroc = roc_auc_score(y_train, y_train_pred_proba)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    precision_scores.append(train_precision)
    recall_scores.append(train_recall)
    auroc_scores.append(train_mean_auroc)
    
    print(f"\n{model_name} Training set:")
    print(f"Mean AUROC: {train_mean_auroc:.4f}")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"Confusion Matrix:\n{train_conf_matrix}")
    print(f"Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    # train on the entire training set
    model.fit(X_train_vec, y_train)
    
    # final test set evaluation
    y_test_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    y_test_pred = model.predict(X_test_vec)
    
    # test set metrics
    test_mean_auroc = roc_auc_score(y_test, y_test_pred_proba)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    precision_scores.append(test_precision)
    recall_scores.append(test_recall)
    auroc_scores.append(test_mean_auroc)
    
    print(f"\n{model_name} Test set:")
    print(f"Mean AUROC: {test_mean_auroc:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Confusion Matrix:\n{test_conf_matrix}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # plot the bar graph
    x = np.arange(len(set_names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_scores, width, label='Precision')
    plt.bar(x, recall_scores, width, label='Recall')
    plt.bar(x + width, auroc_scores, width, label='AUROC')

    plt.xlabel('Sets')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and AUROC by Bernoulli NB Sets')
    plt.xticks(x, set_names)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    saveBestModel(model)

def run_clean_data(best_model, best_model_name, best_model_params):
    # running the larger dataset of 72134 records
    path = "WELFake_Dataset.csv"
    df = pd.read_csv(path)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['combined_text'] = df['title'] + ' ' + df['text']
    # clean version of the data already exists in WELFake_Dataset_Cleaned.csv
    #df['combined_text'] = df['combined_text'].apply(clean_text)  
    cleaned_path = "WELFake_Dataset_Cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    X = df['combined_text']
    Y = df['label']
    train_model(X, Y, best_model, best_model_name, best_model_params)

def run_unclean_data(best_model, best_model_name, best_model_params):
    # running the larger dataset of 72134 records
    path = "WELFake_Dataset.csv"
    df = pd.read_csv(path)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['combined_text'] = df['title'] + ' ' + df['text']
    X = df['combined_text']
    Y = df['label']
    train_model(X, Y, best_model, best_model_name, best_model_params)

def main():
    #running the smaller sample of the data to find out which NB model to use
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(clean_text)
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text']
    X = df['combined_text']
    Y = df['label']
    best_model, best_model_name, best_model_params = find_best_nb_model(X, Y)
    #run_clean_data(best_model, best_model_name, best_model_params) this made no difference in the results :(
    run_unclean_data(best_model, best_model_name, best_model_params)

if __name__=="__main__":
    main()



