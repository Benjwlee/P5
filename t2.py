import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.linear_model import Lars
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.drop('TOTAL', axis=0)

df = df.replace('NaN', 0)
df['email_address'] = df['email_address'].fillna(0).apply(lambda x: x != 0, 1)
#del df['email_address']

df_original = df.copy()

# Convert features to floats since MinMaxScaler does not like int64's
X_original = df.drop(['poi'], axis=1).astype(float)
y_original = df['poi']

# Drop any row that has only zeros in it, drop from labels first, then from features
y_original = y_original[X_original.abs().sum(axis=1) != 0]
X_original = X_original[X_original.abs().sum(axis=1) != 0]

# Save the names of the features 
X_names = X_original.columns
#X_original = X_original.apply(lambda x: x.fillna(0), axis=0)

# Scale the features
standardized = MinMaxScaler().fit_transform(X_original)

# Score the features using a classification scoring function using 
# the Anova F-value for the provided sample
selection = SelectKBest(k='all', score_func=f_classif).fit(standardized, y_original)

#new_X = selection.transform(standardized)

#KBestNames = X_names[selection.get_support()]

# Create a pd.DataFrame of the names and scores
scores = pd.DataFrame([X_names, selection.scores_])
scores = scores.T
scores.columns = ['Features', 'Scores']
scores = scores.sort(['Scores'], ascending=False).reset_index(drop=True)
topKBest = list(scores.Features[0:17])

ET_selection = ExtraTreesClassifier(n_estimators=1000).fit(standardized, y_original)
#print ET_selection.feature_importances_

ET_new_X = selection.transform(standardized)

# Create a pd.DataFrame of the names and importances
scores = pd.DataFrame(ET_selection.feature_importances_, index=X_names)
#scores = scores.T

scores.columns = ['Importance']
scores = scores.sort(['Importance'], ascending=False)
print "TOP10: \n", list(scores.index[0:9])
print scores
scores.sort(['Importance'], ascending=True).plot(kind='barh')
low_var_remover = VarianceThreshold(threshold=.5)
df_50 = df.dropna(axis=1, thresh=len(df)/2)
df_50 = df_50.dropna(axis=0, thresh=5)
df_50.info()
df = df.apply(lambda x: x.fillna(0), axis=0)

import seaborn as sns
sns.set(style='darkgrid')

f, ax = plt.subplots(figsize=(14, 14))
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.corrplot(df.corr(), annot=True, sig_stars=False,
             diag_names=False, cmap=cmap, ax=ax)
f.tight_layout()
corrs = df.corr()
corrs.sort(['poi'], ascending=False)['poi']
df_50.corr().ix[: ,'salary']
cols1 = ['salary', 'other', 'total_stock_value', 'exercised_stock_options', 'total_payments', 'restricted_stock']
cols2= ['salary', 'other', 'total_stock_value', 'exercised_stock_options',  'total_payments', 'restricted_stock', 'bonus']
cols3 = ['to_messages', 'from_this_person_to_poi', 'from_messages',  'shared_receipt_with_poi', 'from_poi_to_this_person']
def kcluster_null(df=None, cols=None, process_all=True):
    '''
    Input: Takes pandas dataframe with values to impute, and a list of columns to impute
        and use for imputing.
    Returns: Pandas dataframe with null values imputed for list of columns passed in.
    
    # Ideally columns should be somewhat correlated since they will be used in KNN to
    # predict each other, one column at a time.
    
    '''
    # Create a KNN regression estimator for 
    income_imputer = KNeighborsRegressor(n_neighbors=1)
    # Loops through the columns passed in to impute each one sequentially.
    if not process_all:
        to_pred = cols[0]
        predictor_cols = cols[1:]

    for each in cols:
        # Create a temp list that does not include the column being predicted.
        temp_cols = [col for col in cols if col != each]
        # Create a dataframe that contains no missing values in the columns being predicted.
        # This will be used to train the KNN estimator.
        df_col = df[df[each].isnull()==False]
        
        # Create a dataframe with all of the nulls in the column being predicted.
        df_null_col = df[df[each].isnull()==True]
        
        # Create a temp dataframe filling in the medians for each column being used to
        # predict that is missing values.
        # This step is needed since we have so many missing values distributed through 
        # all of the columns.
        temp_df_medians = df_col[temp_cols].apply(lambda x: x.fillna(x.median()), axis=0)
        
        # Fit our KNN imputer to this dataframe now that we have values for every column.
        income_imputer.fit(temp_df_medians, df_col[each])
        
        # Fill the df (that has null values being predicted) with medians in the other
        # columns not being predicted.
        # ** This currently uses its own medians and should ideally use the predictor df's
        # ** median values to fill in NA's of columns being used to predict.
        temp_null_medians = df_null_col[temp_cols].apply(lambda x: x.fillna(x.median()), axis=0)
        
        # Predict the null values for the current 'each' variable.
        new_values = income_imputer.predict(temp_null_medians[temp_cols])

        # Replace the null values of the original null dataframe with the predicted values.
        df_null_col[each] = new_values
        
        # Append the new predicted nulls dataframe to the dataframe which containined
        # no null values.
        # Overwrite the original df with this one containing predicted columns. 
        # Index order will not be preserved since it is rearranging each time by 
        # null values.
        df = df_col.append(df_null_col)
        
    # Returned final dataframe sorted by the index names.
    return df.sort_index(axis=0)
df.irow(127)
cols = [x for x in df.columns]
for each in cols:
    g = sns.FacetGrid(df, col='poi', margin_titles=True, size=6)
    g.map(plt.hist, each, color='steelblue')
from pandas.tools.plotting import scatter_matrix
list(df.columns)
financial_cols = np.array(['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value','expenses', 'loan_advances', 'other', 'director_fees', 'deferred_income', 'long_term_incentive'])
email_cols = np.array(['from_messages', 'to_messages', 'shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person', 'email_address'])
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(df[financial_cols], df['poi'])     
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)
padding = np.arange(len(financial_cols)) + 0.5
plt.figure(figsize=(14, 12))
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, financial_cols[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(df[email_cols], df['poi'])

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(email_cols)) + 0.5
plt.figure(figsize=(14, 12))
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, email_cols[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()
all_cols = np.concatenate([email_cols, financial_cols])
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(df[all_cols], df['poi'])

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(all_cols)) + 0.5
plt.figure(figsize=(14, 12))
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, all_cols[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()
df['ex_stock_bins'] = pd.cut(df.exercised_stock_options, bins=15, labels=False)
pd.value_counts(df.ex_stock_bins)
df.exercised_stock_options.plot()
def capValues(x, cap):
    return (cap if x > cap else x)
df.exercised_stock_options = df.exercised_stock_options.apply(lambda x: capValues(x, 5000000))
df['ex_stock_bins'] = pd.cut(df.exercised_stock_options, bins=15, labels=False)
pd.value_counts(df.ex_stock_bins)
df[['ex_stock_bins', 'poi']].groupby('ex_stock_bins').mean().plot()
df.columns
df[['bonus', 'poi']].groupby('bonus').mean().plot()
df.shared_receipt_with_poi.plot()
max(df.shared_receipt_with_poi)
my_bins = [min(df.shared_receipt_with_poi)] + [250] + range(500, 5000, 500) + [max(df.shared_receipt_with_poi)]
df['shared_poi_bins'] = pd.cut(df.shared_receipt_with_poi, bins=my_bins, labels=False, include_lowest=True)
pd.value_counts(df['shared_poi_bins'])
df[['shared_poi_bins', 'poi']].groupby('shared_poi_bins').mean().plot()
df.total_stock_value
from sklearn.preprocessing import StandardScaler

df['total_stock_scaled'] = StandardScaler().fit_transform(df[['total_stock_value']])
df['bonus_scaled'] = StandardScaler().fit_transform(df[['bonus']])
def dont_neg_log(x):
    if x >=0:
        return np.log1p(x)
    else:
        return 0
    
df['stock_log'] = df['total_stock_value'].apply(lambda x: dont_neg_log(x))
financial_cols = np.array(['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
                  'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
                  'expenses', 'loan_advances', 'other', 'director_fees', 'deferred_income', 
                  'long_term_incentive'])

email_cols = np.array(['from_messages', 'to_messages', 'shared_receipt_with_poi', 
              'from_this_person_to_poi', 'from_poi_to_this_person', 'email_address'])
payment_comp = ['salary', 'deferral_payments','bonus', 'expenses', 'loan_advances',
                'other', 'director_fees', 'deferred_income', 'long_term_incentive']
payment_total = ['total_payments']

stock_comp = ['exercised_stock_options', 'restricted_stock','restricted_stock_deferred',]
stock_total = ['total_stock_value']

all_comp = payment_comp + stock_comp

email_comp = ['shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person' ]
email_totals = ['from_messages', 'to_messages'] # interaction_w_poi = total(from/to/shared poi)
df['total_compensation'] = df['total_payments'] + df['total_stock_value']

for each in payment_comp:
    df['{0}_{1}_ratio'.format(each, 'total_pay')] = df[each]/df['total_payments']

for each in stock_comp:
    df['{0}_{1}_ratio'.format(each, 'total_stock')] = df[each]/df['total_stock_value']

for each in all_comp:
    df['{0}_{1}_ratio'.format(each, 'total_compensation')] = df[each]/df['total_compensation']

    
df['total_poi_interaction'] = df['shared_receipt_with_poi'] + df['from_this_person_to_poi'] + \
df['from_poi_to_this_person']

for each in email_comp:
    df['{0}_{1}_ratio'.format(each, 'total_poi_int')] = df[each]/df['total_poi_interaction']

df['total_active_poi_interaction'] = df['from_this_person_to_poi'] + df['from_poi_to_this_person']
df['to_poi_total_active_poi_ratio'] = df['from_this_person_to_poi']/df['total_active_poi_interaction']
df['from_poi_total_active_poi_ratio'] = df['from_poi_to_this_person']/df['total_active_poi_interaction']

df['to_messages_to_poi_ratio'] = df['from_this_person_to_poi']/ df['to_messages']
df['from_messages_from_poi_ratio'] = df['from_poi_to_this_person']/df['from_messages']
df['shared_poi_from_messages_ratio'] = df['shared_receipt_with_poi']/df['from_messages']
df['shared_poi_total_compensation'] = df['shared_receipt_with_poi']/df['total_compensation']
df['bonus_by_total_stock'] = df['bonus']/df['total_stock_value']

## Add squared features
for each in all_comp:
    df['{0}_squared'.format(each)] = df[each]**2
    
for each in email_comp:
    df['{0}_squared'.format(each)] = df[each]**2

df = df.apply(lambda x: x.fillna(0), axis=0)

df[['poi', 'director_fees_total_pay_ratio', 'director_fees', 'total_payments']]

df[df['poi']==True]
df = df.replace([np.inf, -np.inf], 0)
df.ix[20:30, 30:40]
df.ix[11,:]
all_cols2 = np.concatenate([all_cols, np.array(['shared_poi_bins', 'ex_stock_bins', 
                                                'total_stock_scaled', 'bonus_scaled',
                                                'stock_log'])])
# from_messages_from_poi_to_ratio

features = np.array(df.drop('poi', axis=1).columns)

clf = ExtraTreesClassifier(n_estimators=3000)
clf.fit(df[features], df['poi'])

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
plt.figure(figsize=(16,14))
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, features[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

top10_features_RF = ['bonus', 'total_stock_value', 'other', 'total_compensation', 'expenses',
                 'other_total_pay_ratio', 'from_messages_from_poi_ratio', 'restricted_stock',
                 'shared_poi_from_messages_ratio', 'total_payments']

top10_features_ET = ['exercised_stock_options_squared', 'total_stock_value', 'bonus_total_pay_ratio', 
                     'long_term_incentive_total_pay_ratio', 'bonus', 'deferred_income',
                     'total_compensation', 'to_messages_to_poi_ratio',
                     'from_messages_from_poi_ratio', 'to_messages_to_poi_ratio', 'other_total_pay_ratio',
                     'salary_squared', 'other']
confusion_matrix(df['poi'], clf.predict(df[features]))
X_df = df.drop('poi', axis=1)
y_df = df['poi']
selector = SelectKBest(k=12, score_func=f_classif)
selector = selector.fit_transform(X_df, y_df)
selector
class ColumnExtractor(TransformerMixin):
    '''
    Columns extractor transformer for sklearn pipelines.
    Inherits fit_transform() from TransformerMixin, but this is explicitly
    defined here for clarity.
    
    Methods to extract pandas dataframe columns are defined for this class.
    
    '''
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
    def transform(self, X, **transform_params):
        '''
        Input: A pandas dataframe and a list of column names to extract.
        Output: A pandas dataframe containing only the columns of the names passed in.
        '''
        return X[self.columns]
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        return self
top10_features_ET
#X_df = df[['total_payments', 'total_stock_value', 'shared_receipt_with_poi', 'bonus']].astype(float)
X_df = df.drop('poi', axis=1).astype(float)
#X_df = df[top10_features_ET]

#X_df = df[topKBest].astype(float)
y_df = df['poi']

y_df = y_df[X_df.abs().sum(axis=1) != 0]
X_df = X_df[X_df.abs().sum(axis=1) != 0]





sk_fold = StratifiedShuffleSplit(y_df, n_iter=100, test_size=0.1) 
        
pipeline = Pipeline(steps=[#('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy="median", verbose=0)),
                           #('standardizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
                           ('minmaxer', MinMaxScaler()),
                           #('low_var_remover', VarianceThreshold()),
                           ('selection', SelectKBest(score_func=f_classif)),
                           ('reducer', PCA()),
                           #('classifier', LinearSVC(penalty='l1', dual=False)),
                           #('KMeans', KMeans(n_clusters=2))
                           ('classifier', LogisticRegression())
                           #('classifier2', SGDClassifier(n_iter=300))
                                                     ]) # ,
                           #('ET', ExtraTreesClassifier(bootstrap=True, compute_importances=None,
                           #                            criterion='gini', n_estimators=1500, n_jobs=1,
                           #                            oob_score=True, random_state=None, verbose=0,
                           #                            max_features='auto', min_samples_split=2,
                           #                            min_samples_leaf=1))])

                    
params = {
          #'ET__n_estimators': [1500],
          #'ET__max_features': ['auto', None, 3, 5, 10, 20],
          #'ET__min_samples_split': [2, 4, 10],
          #'ET__min_samples_leaf': [1, 2, 5],
          'selection__k': [20, 17, 15],
          'classifier__C': [1, 10, 100, 1000],
          #'classifier2__alpha': [0.0001, 0.001],
          #'classifier2__loss': ['hinge', 'log', 'modified_huber'],
          #'classifier2__class_weight': [{True: 4, False: 1}, {True: 10, False: 1}],
          #'classifier__penalty': ['l1', 'l2'],
          'classifier__class_weight': [{True: 12, False: 1}, {True: 10, False: 1}, {True: 8, False: 1}],
          'classifier__tol': [1e-1, 1e-2, 1e-4, 1e-8, 1e-16, 1e-32],
          'reducer__n_components': [1, 2, 3, 4, 5],
          'reducer__whiten': [True, False]
          #'feature_selection__k': [3, 5, 10, 20]
          #'ET__criterion' : ['gini', 'entropy'],
          #'imputer__strategy': ['median', 'mean'],
          #'low_var_remover__threshold': [0, 0.1, .25, .50, .75, .90, .99]
          }
# Scoring: average_precision, roc_auc, f1, recall, precision
grid_search = GridSearchCV(pipeline, param_grid=params, cv=sk_fold, n_jobs = 1, scoring='f1')
grid_search.fit(X_df, y=y_df)
#test_pred = grid_search.predict(X_test)
#print "Cross_Val_score: ", cross_val_score(grid_search, X_train, y_train)
print "Best Estimator: ", grid_search.best_estimator_
    #f1_avg.append(f1_score(y_test, test_pred))
#print "F1: ", f1_score(y_test, test_pred)
#print "Confusion Matrix: "
#print confusion_matrix(y_test, test_pred)
#print "Accuracy Score: ", accuracy_score(y_test, test_pred)
print "Best Params: ", grid_search.best_params_

n_iter = 1000
sk_fold = StratifiedShuffleSplit(y_df, n_iter=n_iter, test_size=0.1)
f1_avg = []
recall_avg = []
precision_avg = []
for i, all_index in enumerate(sk_fold):
    train_index = all_index[0]
    test_index = all_index[1]
    X_train, X_test = X_df.irow(train_index), X_df.irow(test_index)
    y_train, y_test = y_df[train_index], y_df[test_index]

    grid_search.best_estimator_.fit(X_train, y=y_train)
    # pipeline.fit(X_train, y=y_train)
    test_pred = grid_search.predict(X_test)
    #test_pred = pipeline.predict(X_test)

    #print "Cross_Val_score: ", cross_val_score(grid_search, X_train, y_train)
    #print "Best Estimator: ", grid_search.best_estimator_
    #print f1_score(y_test, test_pred)
    if i % round(n_iter/10) == 0:
        sys.stdout.write('{0}%..'.format(float(i)/n_iter*100)) 
        sys.stdout.flush()        
    f1_avg.append(f1_score(y_test, test_pred))
    precision_avg.append(precision_score(y_test, test_pred))
    recall_avg.append(recall_score(y_test, test_pred))

print "Done!"
print ""
print "F1 Avg: ", sum(f1_avg)/n_iter
print "Precision Avg: ", sum(precision_avg)/n_iter
print "Recall Avg: ", sum(recall_avg)/n_iter

pipeline = Pipeline(steps=[#('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)),
                           #('standardizer', StandardScaler(copy=True, with_mean=True, with_std=True)),
                           #('low_var_remover', VarianceThreshold(threshold=0.1)), 
                           #('feature_selection', LinearSVC()),
                           ('features', FeatureUnion([
                                ('financial', Pipeline([
                                    ('extract', ColumnExtractor(FINANCIAL_FIELDS)),
                                    ('scale', StandardScaler()),
                                    ('reduce', LinearSVC())
                                ])),

                                ('email', Pipeline([
                                    ('extract2', ColumnExtractor(EMAIL_FIELDS)),
                                    ('scale2', StandardScaler()),
                                    ('reduce2', LinearSVC())
                                ]))

                            ])),
                           ('ET', ExtraTreesClassifier(bootstrap=True, compute_importances=None,
                                                       criterion='gini', n_estimators=1500, n_jobs=1,
                                                       oob_score=True, random_state=None, verbose=0,
                                                       max_features=None, min_samples_split=2,
                                                       min_samples_leaf=1))
                            ])

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def test_classifier(clf, dataset, feature_list, folds = 1000):
    #data = featureFormat(dataset, feature_list, sort_keys = True)
    #labels, features = targetFeatureSplit(data)
    labels = y_df
    features = X_df
    cv = StratifiedShuffleSplit(labels, n_iter=folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        #for ii in train_idx:
        #    features_train.append( features[ii] )
        #    labels_train.append( labels[ii] )
        #for jj in test_idx:
        #    features_test.append( features[jj] )
        #    labels_test.append( labels[jj] )
        features_train, features_test = features.irow(train_index), features.irow(test_index)
        labels_train, labels_test = labels[train_index], labels[test_index]
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives

        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        
        precision = 1.0*true_positives/(true_positives+false_positives)
        
        recall = 1.0*true_positives/(true_positives+false_negatives)
        
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
       
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

        print clf
        print ""
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)

        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf

clf = Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), 
                      ('reducer', PCA(copy=True, n_components=4, whiten=True)), 
                      ('classifier', LogisticRegression(C=10, class_weight='auto',
                                                        dual=False, fit_intercept=True,
                                                        intercept_scaling=1, penalty='l2',
                                                        random_state=None, tol=0.0001))])

test_classifier(grid_search.best_estimator_, None, None, folds=1000)












