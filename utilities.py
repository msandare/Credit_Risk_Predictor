# This utilities file will contain a number of useful functions that can be used in this ML/ Data Science Project
import pandas as pd
import numpy as np
#matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# imports for data preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

#supress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')


#################
#function to examine missing values from dataset
def missing_values_table(df):
    #Total missing Values
    mis_val = df.isnull().sum()

    #Percentage missing values
    mis_val_percent = 100* mis_val / len(df)

    #make table with results
    mis_val_table = pd.concat([mis_val, mis_val_percent])

    #rename the columns
    mis_val_table_col_rename = mis_val_table.rename(columns= {0: 'Missing Values', 1: '% of Total Values'})

    #sort table by percentage of missing values decending
    mis_val_table_col_rename = mis_val_table_col_rename[mis_val_table_col_rename.iloc[:,1]!=0].sort_values('% of Total Values', 
    ascending = False).round(1)

    #print summary info
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" + 'There are ' 
    + str(mis_val_table_col_rename.shape[0]) + " columns that have missing values")

    # Return the dataframe with missing information
    return mis_val_table_col_rename


###################
#Plots the kde distribution of a variable colored by the value of the target
def kde_target(var_name, df):
    avg_not_repaid = 0 #placeholder value
    avg_repaid = 0 #placeholder value

    #calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])

    #calcualte the medians for repaid vs not repaid
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    #plot the distribution for target == 0 and target =1
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    plt.legend()

    #chart labels
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()

    #print out the correlation 
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))

    #print out the average value
    print('Median value for the loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid = %0.4f' % avg_repaid)


#########################
#Function to calculate correlations with the a target col for a dataframe
def target_corrs(target_var_name, df):
    #list of correlations
    corrs = []

    #Iterate through the columns 
    for col in df.columns: 
        print(col)
        #skip the target column
        if col != target_var_name:
            #calculare correlation with the target
            corr = df[target_var_name].corr(df[col])

            #append the list as a tuple
            corrs.append((col, corr))

    #sort by absolute magnitude of correlations
    corrs = sorted(corrs, key = lambda x: abs(x[1]), reverse = True)

    return corrs

############################
# Function for Label Encoding
def label_encoder(df):
    le = LabelEncoder()
    le_count = 0

    #iterate through columns 
    for col in df:
        print(col)
        if df[col].dtype == 'object':
            #if two or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                #train on df
                le.fit(df[col])
                #transforms df
                df[col] = le.transform(df[col])

                #keep count of the number of cols that are label encoded
                le_count += 1

    print('%d columns were label encoded.' % le_count) 

    return df


############################
# Function for One-Hot Encoding
def one_hot_encoder(df):
    df = pd.get_dummies(df)
    
    return df


#############################
# Aligning Training and Testing Datasets
def dataframe_alignment(train_df, test_df, train_label):
    #ensure train label is a string
    train_label = str(train_label)

    training_target_labels = train_df[train_label]

    #aligning the training and testing df's, keep only cols in both dfs
    train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)

    train_df[train_label] = training_target_labels

    print('Training Features Shape: ', train_df.shape)
    print('Testing Features Shape: ', test_df.shape)

    return train_df, test_df

###########################
#Aggregates the numeric values in a dataframe. This can be used to create features for each instance of the grouping variable.
'''
Parameters
--------------
    df(dataframe): the dataframe to calculate the statistics on 
    group_var (string): the variable by which to group df
    df_name (string): the variable used to rename the columns

Return
-------------
    agg (dataframe): a dataframe with the statistics aggregated for all numeric columns. Each instance of the grouping variable will have the statistics (mean, min, max, sum; currently supported) calculated. The columns are also renamed to keep track of features created.
'''

def agg_numeric(df, group_var, df_name):
    #Remove id variables other than grouping variables
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    #group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count','mean','max', 'min','sum']).reset_index()

    #need to create new column names
    columns = [group_var]

    #Iterate through the variable names
    for var in agg.columns.levels[0]:
        #skip the grouping variable
        if var != group_var:
            #iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                #make a new colunm name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    agg.columns = columns
    return agg


###########################################
#computes counts and normalized counts for each observation of 'group_var' of each unique category in every categorical variables
'''
Parameters
-------------------
df: dataframe -- the dataframe to calculate the value counts for 
group_var: string -- the variable by which to group the dataframe. For each unique value of this variable, the fina; datafram will have on row
df_name: string -- variable added to the front of columns names to keep track of columns

Return
------------------
categorical: dataframe --- a dataframe with counts and normalized counts of each unique category in every categorical variable with one row for every unique value of the 'group_var'
'''
def count_categorical(df, group_var, df_name):
    #ensure group var is a string
    group_var = str(group_var)
    
    #one hot encoding categorical variables in df
    #select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    #make to put the indentifying id on the column
    categorical[group_var] = df[group_var]
    #Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    columns_names = []

    #removing multi-level indexing (putting all columns on single level)
    #iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        #iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            #make a new column name
            columns_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = columns_names

    return categorical
    

##############################################
#calculating the depreciation of a toyota corolla in Kazakhstan
'''assumptions are:
1) A new Toyota Corolla in Almaty costs 8118662
2) 20% depreciation after first year of ownership
3) 10% depreciation every year after first
'''
def general_car_depreciation(df, new_car_value):
    car_values_list = []

    count = 0
    for row in df['FLAG_OWN_CAR']:
        if row == 0:
            car_values_list.append(0)
        elif row == 1:
            age = df.iloc[count]['OWN_CAR_AGE']
            age.astype(np.int64)

            if np.isnan(age):
                age = 0

            car = new_car_value
            for yr in range(int(age)):
                if yr in range(int(age)):
                    if yr == 0:
                        #assumption car will depreciate 20% in first year
                        car = car - (car * 0.2)
                    else:
                        #assumption car will depreciate 10% every year after first
                        car = car - (car * 0.1)
            car_values_list.append(car)
        count += 1

    return car_values_list



