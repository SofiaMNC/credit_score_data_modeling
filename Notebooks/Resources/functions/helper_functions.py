'''
    This module defines a set of calculations
    functions for project 7.
'''

import time
import pandas as pd
import numpy as np
import sys
import random
import gc
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import load

#------------------------------------------

def get_missing_values_percent_per(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column

        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed

        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''

    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']
    missing_percent_df['Total'] = 100

    return missing_percent_df

#------------------------------------------

def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data))
        files_nb_columns.append(len(file_data.columns))

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def agg_numeric(df, group_var, df_name):
    '''
        Aggregates the numeric values in a dataframe. This can
        be used to create features for each instance of the grouping variable.
        
        Parameters
        --------
        - df        : pandas dataframe
                    The dataframe to calculate the statistics on
        - group_var : string
                    The variable by which to group df
        - df_name   : string
                    The variable used to rename the columns
            
        Return
        --------
        - agg : pandas dataframe
                A dataframe with the statistics aggregated for
                all numeric columns. Each instance of the grouping variable will have
                the statistics (mean, min, max, sum; currently supported) calculated.
                The columns are also renamed to keep track of features created.
    '''

    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns

    return agg

#------------------------------------------

def agg_categorical(df, group_var, df_name):
    '''
        Aggregates the categorical features in a child dataframe
        for each observation of the parent variable.
        
        Parameters
        --------
        - df         : pandas dataframe
                    The dataframe to calculate the value counts for.
            
        - parent_var : string
                    The variable by which to group and aggregate 
                    the dataframe. For each unique value of this variable, 
                    the final dataframe will have one row
            
        - df_name    : string
                    Variable added to the front of column names 
                    to keep track of columns

        Return
        --------
        categorical : pandas dataframe
                    A dataframe with aggregated statistics for each observation 
                    of the parent_var
                    The columns are also renamed and columns with duplicate values 
                    are removed.
    '''
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical

#------------------------------------------

def convert_types(df, print_info = False):
    '''
        Optimized dataframe size by converting the types.

        Parameters
        --------
        - df         : pandas dataframe
                       The dataframe to optimize
        - print_info : boolean
                       Verbose flag
        Return
        --------
        df : pandas dataframe
             The optimized dataframe with converted types
    '''
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df

#------------------------------------------

def remove_missing_columns(df, threshold = 90):
    '''
        Remove columns from dataframe with a percentage
        of missing values > threshold.

        Parameters
        --------
        - df        : pandas dataframe
                      The dataframe to remove columns from.
        - threshold : int 
                      The threshold for the % of missing values
                      for a given feature above which the feature
                      will be deleted.

        Return
        --------
        df : pandas dataframe
             The dataframe without the columns with a % of missing columns
             > threshold.
    '''

    # Calculate missing stats for train and test (remember to calculate a percent!)
    df_miss = pd.DataFrame(df.isnull().sum())
    df_miss['percent'] = 100 * df_miss[0] / len(df)
    
    # list of missing columns for train and test
    missing_columns = list(df_miss.index[df_miss['percent'] > threshold])
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' \
          % (len(missing_columns), threshold))
    
    # Drop the missing columns and return
    df = df.drop(columns = missing_columns)
    
    return df

#------------------------------------------

def aggregate_client(df, group_vars, df_names):
    '''
        Aggregate a dataframe with data at the loan level
        at the client level
    
        Parameters
        --------
        - df : pandas dataframe
               Data at the loan level
        - group_vars : list of two strings
                       grouping variables for the loan, then the client
                    (example ['SK_ID_PREV', 'SK_ID_CURR'])
        - names : list of two strings
                   names to call the resulting columns
                   (example ['cash', 'client'])
            
        Returns
        --------
        - df_client : pandas dataframe
                      Aggregated numeric stats at the client level.
                      Each client will have a single row with all the 
                      numeric data aggregated
    '''
    
    # Aggregate the numeric columns
    df_agg = agg_numeric(df, group_var = group_vars[0], df_name = df_names[0])
    
    # If there are categorical variables
    if any(df.dtypes == 'category'):
    
        # Count the categorical columns
        df_counts = agg_categorical(df, group_var = group_vars[0], df_name = df_names[0])

        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on = group_vars[0], how = 'outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, group_var = group_vars[1], df_name = df_names[1])

        
    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
        
        gc.enable()
        del df_agg
        gc.collect()
        
        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns = [group_vars[0]])
        
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, group_var = group_vars[1], df_name = df_names[1])
        
    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client

#------------------------------------------

def bank_score(y_true, y_pred):
    '''
        Cost function to minimize the risk for the
        loaning firm.
    
        Parameters
        --------
        - y_true : pandas dataframe
                   The true values of the target
        - y_pred : pandas dataframe
                   The predicted values for the target
            
        Returns
        --------
        - score : float
                  The score for the prediction
    '''

    (TN, FP, FN, TP) = confusion_matrix(y_true, y_pred).ravel()
    N = TN + FP    # total negatives cases
    P = TP + FN    # total positives cases
    
    # Setting the bank's gain and loss for each case
    FN_value = -10   # The loan is granted but the customer defaults : the bank loses money (Type-II Error)
    TN_value = 1     # The loan is reimbursed : the bank makes money
    TP_value = 0     # The loan is (rightly) refused : the bank neither wins nor loses money
    FP_value = -1    # Loan is refused by mistake : the bank loses money it could have made, 
                     # but does not actually lose any money (Type-I Error)

    # calculate total gains
    gain = TP*TP_value + TN*TN_value + FP*FP_value + FN*FN_value
    
    # best score : all observations are correctly predicted
    best = N*TN_value + P*TP_value 

    # baseline : all observations are predicted = 0
    baseline = N*TN_value + P*FN_value 
    
    # normalize to get score between 0 (baseline) and 1
    score = (gain - baseline) / (best - baseline)
    
    return score

#------------------------------------------

def cv_score_sample(model, x, y, x_test, y_test, scoring, folds=5, loss_func=None):
    '''
        Uses cross-validation to determine the score of a model 
        on train data, then calculates the score on test data.
    
        Parameters
        --------
        - model     : a machine learning model
        - x         : pandas dataframe
                      The training features
        - y         : pandas dataframe
                      The training labels
        - x_test    : pandas dataframe
                      The test features
        - y_test    : pandas dataframe
                      The test labels
        - scoring   : Cost function
                      The cost function to use for scoring
        - folds     : int
                      The number of folds to use for the cross-validation
        - loss_func : Loss function
                      The loss function to use for the algorithms that allow
                      custom loss functions
            
        Returns
        --------
        -, -, -, - : tuple
                     - The training custom scores for each fold (array)
                     - The custom score for the test data (float)
                     - The training ROC AUC scores for each fold (array)
                     - The ROC AUC score for the test data (float)
    '''

    cv_custom_scores = []
    cv_ra_scores = []

    y_pred_proba = []

    # create folds
    kf = StratifiedKFold(n_splits=folds)
    
    for train_indices, valid_indices in kf.split(x, y):
        # Training data for the fold
        xtrn, ytrn = x.iloc[train_indices], y.iloc[train_indices]
        # Validation data for the fold
        xval, yval = x.iloc[valid_indices], y.iloc[valid_indices]

        # train
        if loss_func!=None:
            model.fit(xtrn, ytrn, eval_metric = loss_func)
        else:
            model.fit(xtrn, ytrn)

        # predict values on validation set
        ypred = model.predict(xval)
        
        # save probabilities for class 1
        yprob = model.predict_proba(xval)
        y_pred_proba+=(list(yprob[:,1]))

        # calculate and save scores
        ra_score = round(roc_auc_score(yval, ypred), 3)
        cv_ra_scores.append(ra_score)

        custom_score = round(scoring(yval, ypred), 3)
        cv_custom_scores.append(custom_score)

    if loss_func!=None:
        model.fit(x, y, eval_metric=loss_func)
        y_pred = model.predict(x_test)
    else:
        model.fit(x, y)
        y_pred = model.predict(x_test)

    ra_score_test = round(roc_auc_score(y_test, y_pred), 3)

    custom_score_test = round(scoring(y_test, y_pred), 3)

    return np.array(cv_custom_scores), \
           custom_score_test, \
           np.array(cv_ra_scores), \
           ra_score_test

#------------------------------------------

def identify_zero_importance_features(train, train_labels, iterations = 2):
    '''
        Identify zero importance features in a training dataset based on the 
        feature importances from a gradient boosting model. 
        
        Parameters
        --------
        - train        : pandas dataframe
                          Training features
            
        - train_labels : numpy array
                         Labels for training data
            
        - iterations   : int
                         Number of cross validation splits to use for 
                         determining feature importances

        Returns
        --------
        - zero_features       : list
                                The features with 0.0 importance

        - feature_importances : pandas dataframe
                                The importance of all features
    '''
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
    
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size = 0.25, random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances

#------------------------------------------

def transform_data(app_df, bureau_df, bureau_balance_df,
                   card_df, cash_df, installments_df, prev_app_df):
    
    '''
        Transforms new data to as to be consumed by the credit score 
        prediction model.

        Parameters
        --------
        - app_df            : pandas dataframe
                              Main training data
        - bureau_df         : pandas dataframe
                              Bureau information
        - bureau_balance_df : pandas dataframe
                              Bureau balance information
        - card_df           : pandas dataframe
                              Credit card information
        - cash_df           : pandas dataframe
                              Cash information
        - installments_df   : pandas dataframe
                              Installments information
        - prev_app_df       : pandas dataframe
                              Previous applications informations

        Returns
        --------
        - app_df : pandas dataframe
                   The input data ready for consumption by the model
    '''
    
    # Features métier
    app_df["SELF_FINANCED_PERCENT"] = (app_df["AMT_GOODS_PRICE"] - app_df["AMT_CREDIT"])/app_df["AMT_GOODS_PRICE"]*100
    app_df["SELF_FINANCED_PERCENT"] = app_df["SELF_FINANCED_PERCENT"].map(lambda x: 0 if x<0 else x)
    app_df["ANNUITY_ON_INCOME"] = app_df["AMT_ANNUITY"] / app_df["AMT_INCOME_TOTAL"] * 100
    app_df['DAYS_EMPLOYED_PERCENT'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_BIRTH']
    
    # BUREAU
    bureau_df = convert_types(bureau_df, print_info=True)

    # Creation de previous_loan_counts
    previous_loan_counts = bureau_df.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count()\
                                    .rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})

    # Merge de previous_loan_counts dans train on SK_ID_CURR, left
    app_df = app_df.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

    # fillna(0) dans train
    app_df['previous_loan_counts'] = app_df['previous_loan_counts'].fillna(0)
    
    # Creation de bureau_counts
    bureau_counts = agg_categorical(bureau_df, 
                                    group_var = 'SK_ID_CURR', 
                                    df_name = 'bureau')

    # Creation de bureau_agg
    bureau_agg = agg_numeric(bureau_df.drop(columns = ['SK_ID_BUREAU']),
                             group_var = 'SK_ID_CURR',
                             df_name = 'bureau')

    
    # BUREAU BALANCE
    bureau_balance_df = convert_types(bureau_balance_df, print_info=True)

    # Comptage des catégories
    # Counts of each type of status for each previous loan
    bureau_balance_counts = agg_categorical(bureau_balance_df,
                                              group_var = 'SK_ID_BUREAU',
                                              df_name = 'bureau_balance')


    # Creation de bureau_balance_agg
    # Calculate value count statistics for each `SK_ID_CURR`
    bureau_balance_agg = agg_numeric(bureau_balance_df,
                                     group_var = 'SK_ID_BUREAU',
                                     df_name = 'bureau_balance')


    # Creation de bureau_by_loan
    # Dataframe grouped by the loan
    bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts,
                                              right_index = True,
                                              left_on = 'SK_ID_BUREAU',
                                              how = 'outer')

    # Merge to include the SK_ID_CURR
    bureau_by_loan = bureau_by_loan.merge(bureau_df[['SK_ID_BUREAU', 'SK_ID_CURR']],
                                          on = 'SK_ID_BUREAU',
                                          how = 'left')


    # Creation de bureau_balance_by_client
    bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']),
                                           group_var = 'SK_ID_CURR',
                                           df_name = 'client')
    
    # Insert computed features into training data :

    # Merge bureau_counts dans app_train
    app_df = app_df.merge(bureau_counts, on="SK_ID_CURR", how="left")

    # Merge bureau_agg dans app_train
    app_df = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Merge bureau_balance_by_client dans app_train
    app_df = app_df.merge(bureau_balance_by_client, on="SK_ID_CURR", how="left")

    # Suppression des colonnes missing
    app_df = remove_missing_columns(app_df)
    
    
    # PREVIOUS APPLICATION
    
    # Convert types de previous
    prev_app_df = convert_types(prev_app_df, print_info=True)

    # Creation de previous_agg
    prev_agg = agg_numeric(prev_app_df, 'SK_ID_CURR', 'previous')

    # Creation de previous_counts
    prev_counts = agg_categorical(prev_app_df, 'SK_ID_CURR', 'previous')
    
    # Merge previous_counts dans app_train
    app_df = app_df.merge(prev_counts, on="SK_ID_CURR", how="left")

    # Merge previous_agg dans app_train
    app_df = app_df.merge(prev_agg, on="SK_ID_CURR", how="left")

    # Suppression des colonnes missing
    app_df = remove_missing_columns(app_df)
    
    
    # CASH
    
    # Convert types de cash
    cash_df = convert_types(cash_df, print_info=True)

    # Creation de cash_by_client
    cash_by_client = aggregate_client(cash_df,
                                      group_vars = ['SK_ID_PREV', 'SK_ID_CURR'],
                                      df_names = ['cash', 'client'])

    # Merge cash_by_client dans app_train
    app_df = app_df.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')

    # Suppression des colonnes missing
    app_df = remove_missing_columns(app_df)
    
    
    # CARD
    
    # Convert types de credit
    card_df = convert_types(card_df, print_info=True)

    # Creation de credit_by_client
    credit_by_client = aggregate_client(card_df,
                                        group_vars = ['SK_ID_PREV', 'SK_ID_CURR'],
                                        df_names = ['credit', 'client'])

    # Merge credit_by_client dans app_train
    app_df = app_df.merge(credit_by_client, on="SK_ID_CURR", how="left")

    # Suppression des colonnes missing
    app_df = remove_missing_columns(app_df)
    
    
    # INSTALLMENTS PAYMENTS

    # Convert types de installments
    installments_df = convert_types(installments_df, print_info = True)

    # Creation de installments_by_clients
    installments_by_client = aggregate_client(installments_df,
                                              group_vars = ['SK_ID_PREV', 'SK_ID_CURR'],
                                              df_names = ['installments', 'client'])

    # Merge installments_by_clietns dans app_train
    app_df = app_df.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')

    # Suppression des colonnes missing
    app_df = remove_missing_columns(app_df)
    
    # Create an anomalous flag column
    app_df['DAYS_EMPLOYED_ANOM'] = app_df["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # LABEL ENCODING AND ONE HOT ENCODING
    
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in app_df:
        if app_df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(app_df[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_df[col])
                # Transform both training and testing data
                app_df[col] = le.transform(app_df[col])

                # Keep track of how many columns were label encoded
                le_count += 1
                
    # one-hot encoding of categorical variables
    app_df = pd.get_dummies(app_df)
    
    # Keeping only relevant columns
    columns_to_keep = load("./Resources/model_features.joblib")

    app_df = app_df.loc[:, app_df.columns.isin(columns_to_keep)]
    
    # Adjust for missing columns
    missing_columns_list = np.setdiff1d(columns_to_keep, app_df.columns)
    missing_values_df = pd.DataFrame(0, index=np.arange(len(app_df)), columns=missing_columns_list)
    
    app_df = pd.concat([app_df, missing_values_df], axis=1)

    return app_df
