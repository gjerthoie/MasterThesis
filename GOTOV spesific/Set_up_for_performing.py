#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from datetime import datetime



# In[4]:


# Function to train, predict, and evaluate the model
def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, print_cm=True, cm_cmap=plt.cm.Greens):
    results = dict()

    train_start_time = datetime.now()
    print('Training the model...')
    model.fit(X_train, y_train)
    print('Done\n')
    train_end_time = datetime.now()
    results['training_time'] = train_end_time - train_start_time
    print(f"Training time (HH:MM:SS.ms): {results['training_time']}\n")

    print('Predicting test data...')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done\n')
    results['testing_time'] = test_end_time - test_start_time
    print(f"Testing time (HH:MM:SS:ms): {results['testing_time']}\n")
    results['predicted'] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print(f'\n    {accuracy}\n')

    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm:
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print(f'\n {cm}')

    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=cm_normalize, title='Normalized confusion matrix', cmap=cm_cmap)
    plt.show()

    print('-------------------------')
    print('| Classification Report |')
    print('-------------------------')
    classification_rep = classification_report(y_test, y_pred, target_names=class_labels)
    results['classification_report'] = classification_rep
    print(classification_rep)

    results['model'] = model
    return results


# In[6]:


# Function to print GridSearchCV attributes
def print_grid_search_attributes(model):
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print(f'\n\t{model.best_estimator_}\n')

    print('--------------------------')
    print('|     Best Parameters     |')
    print('--------------------------')
    print(f'\tParameters of best estimator:\n\n\t{model.best_params_}\n')

    print('---------------------------------')
    print('|   No of Cross Validation Sets  |')
    print('---------------------------------')
    print(f'\n\tTotal number of cross validation sets: {model.cv}\n')

    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print(f'\n\tAverage Cross Validate score of best estimator:\n\n\t{model.best_score_}\n')


# In[8]:




import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Greens):
    """
    Plots a confusion matrix. Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

