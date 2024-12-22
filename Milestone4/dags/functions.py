import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import re
from sqlalchemy import create_engine




def tidy_col_names(df):
    df_temp=df.copy()
    df_temp.columns = df_temp.columns.str.lower().str.replace(' ', '_')
    return df_temp
def choose_suitable_index(df):
    df.set_index('loan_id', inplace=True)
    print('index set succesfully')
    return df

def clean_emp_title(title):
    if isinstance(title, str):  # not null
        return re.sub(r'[^a-z0-9 ]', '', title.strip().lower())

def helper_emp_length(x):
    if pd.isnull(x):
        return x
    else:
        x= x.replace('< 1 year', '0.5 years')
        x= x.replace('10+ years', '10 years')
        return float(x.split()[0])
def clean_emp_length(df):
    df['emp_length'] = df['emp_length'].apply(helper_emp_length)
    return df    
    
def help_clean_term(x):
    return int(x.split()[0])
def clean_term(df):
    df['term'] = df['term'].apply(help_clean_term)
    return df    
def help_issue_date(x):
    return pd.to_datetime(x)
def clean_issue_date(df):
    df['issue_date'] = df['issue_date'].apply(help_issue_date)
    return df    
def help_clean_type(x):
    return x.lower().replace('joint app', 'joint')
def clean_type(df):
    df['type'] = df['type'].apply(help_clean_type)
    return df    
def help_pymnt_plan(x):
    return int(x)
def clean_pymnt_plan(df):
    df['pymnt_plan'] = df['pymnt_plan'].apply(help_pymnt_plan)
    return df

def handle_incosistent_values(df_org):
    df = df_org.copy()
    df['emp_title'] = df['emp_title'].apply(clean_emp_title)
    df=  clean_emp_length(df)
    df = clean_term(df)
    df = clean_issue_date(df)
    df = clean_type(df)
    df = clean_pymnt_plan(df)
    return df    
def fill_missing_annual_inc_joint(df):
    df.loc[df['type'] != 'joint', 'annual_inc_joint'] = df['annual_inc']
    return df
def fill_missing_values_emp(df):
    df['emp_title'].fillna('missing', inplace=True)
    quantiles = df['annual_inc'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])  
    income_bins = quantiles.values  
    income_labels = [f'{int(income_bins[i])}-{int(income_bins[i+1])}K' for i in range(len(income_bins)-1)]
    df['income_group'] = pd.cut(df['annual_inc'], bins=income_bins, labels=income_labels, include_lowest=True)
    median_emp_length = df.groupby('income_group')['emp_length'].median()
    
    df['emp_length'] = df.apply(
        lambda row: median_emp_length[row['income_group']] if pd.isnull(row['emp_length']) else row['emp_length'], axis=1
    )    
    df.drop(columns='income_group', inplace=True)
    return df




def fill_missing_values_int_rate(df):
    mean_int_rate_by_grade = df.groupby('grade')['int_rate'].mean()
    df['int_rate'] = df.apply(
        lambda row: mean_int_rate_by_grade[row['grade']] if pd.isnull(row['int_rate']) else row['int_rate'],
        axis=1
    )
    return df

def fill_missing_values_description(df):
    df['description'].fillna('No description provided', inplace=True)
    return df

def handle_missing_values(df_org):
    df = df_org.copy()
    df = fill_missing_annual_inc_joint(df)
    df = fill_missing_values_emp(df)
    df = fill_missing_values_int_rate(df)
    df = fill_missing_values_description(df)
    return df

def handle_outliers_annual_inc(df):
    df['annual_inc'] = np.log(df['annual_inc'] + 1)  
    return df    
def handle_outliers_annual_inc_joint(df):    
    df['annual_inc_joint'] = np.log(df['annual_inc_joint'] + 1) 
    return df    

def handle_outliers_avg_cur_bal(df):
    Q1 = df['avg_cur_bal'].quantile(0.25)
    Q3 = df['avg_cur_bal'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR,0)
    upper_bound = Q3 + 1.5 * IQR    
    df['avg_cur_bal'] = df['avg_cur_bal'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
            
    return df    
def handle_outliers_tot_cur_bal(df):
    Q1 = df['tot_cur_bal'].quantile(0.25)
    Q3 = df['tot_cur_bal'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR,0)
    upper_bound = Q3 + 1.5 * IQR
    df['tot_cur_bal'] = df['tot_cur_bal'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)) 
    return df    
def handle_outliers_int_rate(df):
    df['zscore'] = stats.zscore(df['int_rate'])
    threshold = 3
    lower_bound = np.percentile(df['int_rate'], 5)
    upper_bound = np.percentile(df['int_rate'], 95)
    df['int_rate'] = df['int_rate'].where(df['zscore'].abs() < threshold, 
                                       np.where(df['zscore'] > 0, upper_bound, lower_bound))
    df.drop(columns='zscore', inplace=True)
    return df   





def handle_outliers(df_org):
    df=df_org.copy()
    df = handle_outliers_annual_inc(df)
    df = handle_outliers_annual_inc_joint(df)
    df = handle_outliers_avg_cur_bal(df)
    df = handle_outliers_tot_cur_bal(df)
    df = handle_outliers_int_rate(df)
    return df     
def add_month_number(df):
    df['month_number'] = df['issue_date'].dt.month
    
    return df   
def add_salary_can_cover(df):
    df['salary_can_cover'] = np.exp(df['annual_inc_joint'])>= df['loan_amount']
    df['salary_can_cover'] = df['salary_can_cover'].astype(int)
   
    return df     
def encode_letter_grade(df):
    bins = [1, 5, 10, 15, 20, 25, 30, 35]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['letter_grade'] = pd.cut(df['grade'], bins=bins, labels=labels, right=True, include_lowest=True)
    df.drop(columns='grade', inplace=True)
    

    return df
def calculate_installment_per_month(df):
    r = df['int_rate'] / 12
    n = df['term']
    P = df['loan_amount']
    df['installment_per_month'] = P *(( r * (1 + r)**n) / ((1 + r)**n - 1))
    
    return df
def add_columns(df_org):
    df = df_org.copy()
    df = add_month_number(df)
    df = add_salary_can_cover(df)
    df = encode_letter_grade(df)
    df = calculate_installment_per_month(df)
    return df

def encode_one_hot(df):
    nominal_vars = ['home_ownership', 'verification_status', 'loan_status', 'type', 'purpose']
    all_encoded_columns = []
    for var in nominal_vars:
        df_encoded = pd.get_dummies(df[var], prefix=var).astype(int)
        df = pd.merge(df, df_encoded, left_index=True, right_index=True)
        # df.drop(columns=var, inplace=True)


    return df    


def encode_ordinal(df):
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['letter_grade'] = pd.Categorical(df['letter_grade'], categories=grade_order, ordered=True)
    df['letter_grade_encoded'] = df['letter_grade'].cat.codes
    # df.drop(columns='letter_grade', inplace=True)
    
    return df    

def grade(df):
    
    df_temp = df.copy()
    old_grade = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    ordinal_grade=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7]
    df_temp['letter_grade_encoded'] = df['grade'].replace(old_grade, ordinal_grade)
    return df
def encode(df_org):
    df = df_org.copy()
    df = encode_one_hot(df)
    df = encode_ordinal(df)
    return df   

def normalize(df_org):
    df=df_org.copy()
    columns_to_standardize = ['avg_cur_bal', 'tot_cur_bal', 'loan_amount', 
                           'funded_amount', 'int_rate', 'installment_per_month']
    new_columns_to_standardize = ['avg_cur_bal_normalized', 'tot_cur_bal_normalized', 'loan_amount_normalized', 
                           'funded_amount_normalized', 'int_rate_normalized', 'installment_per_month_normalized']
    scaler = StandardScaler()
    
    # Fit and transform the selected columns
    df[new_columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df  
# def clean(df):
    
#     df_clean=   tidy_col_names(df)
#     df_clean=   choose_suitable_index(df_clean)
#     df_clean2=handle_incosistent_values(df_clean)
#     df_clean3=handle_missing_values(df_clean2)
#     df_clean4=handle_outliers(df_clean3)
#     df_clean5=add_columns(df_clean4)
#     df_clean6=encode(df_clean5)
#     df_clean7=normalize(df_clean6)
#     # df_clean8=getStateName(df_clean7)    
#     return df_clean7



def extract_clean(filename):
    df = pd.read_csv(filename)
    print('file loaded succesfully')
    df_clean= tidy_col_names(df)
    df_clean= choose_suitable_index(df_clean)
    df_clean2=handle_incosistent_values(df_clean)
    df_clean3=handle_missing_values(df_clean2)
    df_clean4=handle_outliers(df_clean3)
    df_clean5=add_columns(df_clean4)
    # df_clean5.to_csv('/opt/airflow/data/fintech_clean.csv',index=False)

    df_clean5.to_csv('./data/fintech_clean.csv',index=False)
    print('loaded after cleaning succesfully')

def transform(filename):
    df = pd.read_csv(filename)
    df=encode(df)
    df=normalize(df)
    try:
        # df.to_csv('/opt/airflow/data/fintech_transformed.csv',index=False, mode='x')

        df.to_csv('./data/fintech_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')



def load_to_db(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'fintech',con = engine,if_exists='replace')

    