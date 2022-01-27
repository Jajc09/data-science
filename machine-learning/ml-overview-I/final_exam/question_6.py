import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats

url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv"

df = pd.read_csv(url)
df.head()

df.info()
df.shape
# (13599, 15)

# Balanced data? 
df.great_customer_class.value_counts()
# 0    12431
# 1     1168
# Data is imbalanced

# Data cleaning 
def data_cleaning(df):
    df = df.drop(columns="user_id") # drop id 
    # convert all the columns into numeric type by get_dummies
    df_dummy = pd.get_dummies(df.select_dtypes(exclude='number'))
    df = pd.concat([df, df_dummy], axis=1)
    df = df.drop(columns=["workclass", "marital-status", "occupation", "race", "sex"])
    return df

df = data_cleaning(df)

# some objects I need
X = df
y = np.array(df.pop('great_customer_class'))
num_feats = 13

# Data Feature Selection (Pearson)
def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, y, num_feats)
print(str(len(cor_feature)), 'selected features')
# 13 selected features
cor_feature
# ['sex_Female', 'occupation_service', 'workclass_government', 'workclass_private', 'marital-status_Never-married', 
# 'education_rank', 'workclass_self_employed', 'works_hours', 'occupation_professional', 'occupation_tech', 
# 'occupation_executive', 'marital-status_Divorced', 'marital-status_Married']


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

#chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
#print(str(len(chi_feature)), 'selected features')

# Predictions - RANDOM FOREST
# Split Data into Training and Testing Set

#Save 30% for testing
from sklearn.model_selection import train_test_split


X_new = X.loc[:,['sex_Female', 'occupation_service', 'workclass_government', 'workclass_private', 'marital-status_Never-married', 
          'education_rank', 'workclass_self_employed', 'works_hours', 'occupation_professional', 'occupation_tech', 
          'occupation_executive', 'marital-status_Divorced', 'marital-status_Married']]
y

X_train, X_test, y_train, y_test = train_test_split(X_new, y, 
                                                          stratify = y,
                                                          test_size = 0.3, 
                                                          random_state = 30)

train = X_train.fillna(X_train.mean())
test = X_test.fillna(X_train.mean())

features = list(train.columns)

train.shape
test.shape

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=30,
                               n_jobs=-1, 
                               verbose = 1)

fitting = model.fit(train, y_train)
fitting.classes_

train_predictions = fitting.predict(train)
print(train_predictions)
train_pred_prob = fitting.predict_proba(train)[:, 1]
print(train_pred_prob)

test_predictions = fitting.predict(test)
print(test_predictions)
test_pred_prob = fitting.predict_proba(test)[:, 1]
print(test_pred_prob)

n_nodes = []
max_depths = []

for ind_tree in fitting.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
# Average number of nodes 1195
print(f'Average maximum depth {int(np.mean(max_depths))}')
# Average maximum depth 21

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score

print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_pred_prob)}')
# Train ROC AUC Score: 0.9727330894433909
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, test_pred_prob)}')
# Test ROC AUC  Score: 0.8803056300268097
print(f'Accuracy Score: {accuracy_score(y_test, test_predictions)}')
# Accuracy Score: 0.9208333333333333
print(f'Precision Score: {precision_score(y_test, test_predictions)}')
# Accuracy Score: 0.5645933014354066
print(f'Recall Score: {recall_score(y_test, test_predictions)}')
# Accuracy Score: 0.33714285714285713

from sklearn.metrics import confusion_matrix
import itertools

#  Helper function to plot Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=3)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

cm = confusion_matrix(y_test, test_predictions)
plot_confusion_matrix(cm, classes = ['0', '1'], title = '')
# [[3639   91]
# [ 232  118]]

print('Health Confusion Matrix')

# Determine correct metric to evaluate your prediction model
'''
Because class 1 is unbalanced with respect to class 0, it is better to use the recall and precision metrics to know 
if our model is good at detecting the class with fewer elements. The accuracy in this case would not be appropriate
'''

