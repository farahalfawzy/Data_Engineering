import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import re
import requests
from bs4 import BeautifulSoup


state_mapping = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

def tidy_col_names(df):
    df_temp=df.copy()
    df_temp.columns = df_temp.columns.str.lower().str.replace(' ', '_')
    return df_temp
def choose_suitable_index(df):
    df.set_index('loan_id', inplace=True)
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
    df= clean_emp_length(df)
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
    income_bins_df = pd.DataFrame({
        'income_bin_start': income_bins[:-1],
        'income_bin_end': income_bins[1:],
        'median_emp_length': median_emp_length.values
    })
    income_bins_df.to_csv('data/income_bins_median_emp_length.csv', index=False)
    
    df.drop(columns='income_group', inplace=True)
    return df

def fill_missing_values_emp_stream(df):
    df['emp_title'].fillna('missing', inplace=True)
    # Check if df['emp_length'] has null values
    if df['emp_length'].isnull().any():
        income_bins_median_emp_length = pd.read_csv('data/income_bins_median_emp_length.csv')
    
        df['income_group'] = pd.cut(df['annual_inc'], bins=income_bins_median_emp_length['income_bin_start'], labels=False)
        df['income_group'] = df['income_group'].apply(lambda x: x if x >= 0 else np.nan)
        df['emp_length'] = df.apply(
            lambda row: income_bins_median_emp_length['median_emp_length'][row['income_group']] if pd.isnull(row['emp_length']) else row['emp_length'],
            axis=1
        )   
        df.drop(columns='income_group', inplace=True)

    return df

def fill_missing_values_int_rate(df):
    mean_int_rate_by_grade = df.groupby('grade')['int_rate'].mean()
    df['int_rate'] = df.apply(
        lambda row: mean_int_rate_by_grade[row['grade']] if pd.isnull(row['int_rate']) else row['int_rate'],
        axis=1
    )
    mean_int_rate_by_grade = df.groupby('grade', as_index=False)['int_rate'].mean()
    mean_int_rate_by_grade.to_csv('data/mean_int_rate_by_grade.csv', index=False)
    return df

def fill_missing_values_int_rate_stream(df):
    if df['int_rate'].isnull().any():
        mean_int_rate_by_grade = pd.read_csv('data/mean_int_rate_by_grade.csv')
        grade_int_rate_dict = mean_int_rate_by_grade.set_index('grade')['int_rate'].to_dict()

        # Impute missing int_rate using the loaded mean values
        df['int_rate'] = df.apply(
            lambda row: grade_int_rate_dict[row['grade']] if pd.isnull(row['int_rate']) else row['int_rate'],
            axis=1
        )
    return df

def fill_missing_values_description(df):
    df['description'].fillna('No description provided', inplace=True)
    return df
def handle_missing_values_stream(df_org):
    df = df_org.copy()
    df = fill_missing_annual_inc_joint(df)
    df = fill_missing_values_emp_stream(df)
    df = fill_missing_values_int_rate_stream(df)
    df = fill_missing_values_description(df)
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
    new_bounds = pd.DataFrame({
                'column': ['avg_cur_bal'],
                'lower_bound': [lower_bound],
                'upper_bound': [upper_bound]
            })
    if os.path.exists('data/outlier_bounds.csv'):
        new_bounds.to_csv('data/outlier_bounds.csv', mode='a', header=False, index=False)
    else:
        new_bounds.to_csv('data/outlier_bounds.csv', index=False)
    
            
    return df    
def handle_outliers_tot_cur_bal(df):
    Q1 = df['tot_cur_bal'].quantile(0.25)
    Q3 = df['tot_cur_bal'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR,0)
    upper_bound = Q3 + 1.5 * IQR
    df['tot_cur_bal'] = df['tot_cur_bal'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    new_bounds = pd.DataFrame({
                'column': ['tot_cur_bal'],
                'lower_bound': [lower_bound],
                'upper_bound': [upper_bound]
            })
    if os.path.exists('data/outlier_bounds.csv'):
        new_bounds.to_csv('data/outlier_bounds.csv', mode='a', header=False, index=False)
    else:
        new_bounds.to_csv('data/outlier_bounds.csv', index=False)
    return df    
def handle_outliers_int_rate(df):
    df['zscore'] = stats.zscore(df['int_rate'])
    threshold = 3
    lower_bound = np.percentile(df['int_rate'], 5)
    upper_bound = np.percentile(df['int_rate'], 95)
    df['int_rate'] = df['int_rate'].where(df['zscore'].abs() < threshold, 
                                       np.where(df['zscore'] > 0, upper_bound, lower_bound))
    df.drop(columns='zscore', inplace=True)
    new_bounds = pd.DataFrame({
                'column': ['int_rate'],
                'lower_bound': [lower_bound],
                'upper_bound': [upper_bound]
            })
    if os.path.exists('data/outlier_bounds.csv'):
        new_bounds.to_csv('data/outlier_bounds.csv', mode='a', header=False, index=False)
    else:
        new_bounds.to_csv('data/outlier_bounds.csv', index=False)
    return df   

def load_bounds(column_name):
    BOUNDS_FILE = 'data/outlier_bounds.csv'
    """Loads the lower and upper bounds for the specified column from CSV."""
    if not os.path.exists(BOUNDS_FILE):
        raise FileNotFoundError(f"{BOUNDS_FILE} not found. Ensure bounds are pre-calculated.")

    bounds_df = pd.read_csv(BOUNDS_FILE)
    if column_name not in bounds_df['column'].values:
        raise ValueError(f"No bounds found for column {column_name} in {BOUNDS_FILE}.")
    
    lower_bound, upper_bound = bounds_df.loc[bounds_df['column'] == column_name, ['lower_bound', 'upper_bound']].values[0]
    return lower_bound, upper_bound


def handle_outliers_avg_cur_bal_streaming(df):
    column_name = 'avg_cur_bal'
    lower_bound, upper_bound = load_bounds(column_name)
    
    df[column_name] = df[column_name].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df


def handle_outliers_tot_cur_bal_streaming(df):
    column_name = 'tot_cur_bal'
    lower_bound, upper_bound = load_bounds(column_name)
    
    # Apply bounds to streaming data for tot_cur_bal
    df[column_name] = df[column_name].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df


def handle_outliers_int_rate_streaming(df):
    column_name = 'int_rate'
    lower_bound, upper_bound = load_bounds(column_name)

    # Apply bounds to streaming data for int_rate using z-score threshold
    df['zscore'] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    threshold = 3  # Optional: adjust the z-score threshold if needed

    df[column_name] = df[column_name].where(df['zscore'].abs() < threshold,
                                            np.where(df['zscore'] > 0, upper_bound, lower_bound))
    df.drop(columns='zscore', inplace=True)
    return df

def handle_outliers_stream(df_org):
    df=df_org.copy()
    df = handle_outliers_annual_inc(df)
    df = handle_outliers_annual_inc_joint(df)
    df = handle_outliers_avg_cur_bal_streaming(df)
    df = handle_outliers_tot_cur_bal_streaming(df)
    df = handle_outliers_int_rate_streaming(df)
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
        df.drop(columns=var, inplace=True)


    return df    
def encode_one_hot_stream(df_stream):
    # Load the previously saved one-hot encoded column names
    expected_columns = pd.read_csv('data/one_hot_encoded_columns.csv', header=None).squeeze().tolist()
    
    nominal_vars = ['home_ownership', 'verification_status', 'loan_status', 'type', 'purpose']
    for var in nominal_vars:
        df_encoded = pd.get_dummies(df_stream[var], prefix=var).astype(int)
        print(df_encoded)
        df_stream = pd.concat([df_stream, df_encoded], axis=1)
        df_stream.drop(columns=var, inplace=True)

    # Add any missing columns from the saved list with zeros
    for col in expected_columns:
        if col not in df_stream.columns:
            df_stream[col] = 0

    # Ensure the column order matches the expected columns
    df_stream = df_stream.reindex(columns=expected_columns, fill_value=0)
    
    return df_stream

def encode_ordinal(df):
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['letter_grade'] = pd.Categorical(df['letter_grade'], categories=grade_order, ordered=True)
    df['letter_grade_encoded'] = df['letter_grade'].cat.codes
    df.drop(columns='letter_grade', inplace=True)
    
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
    scaler = StandardScaler()
    
    # Fit and transform the selected columns
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    normalization_params = pd.DataFrame({
        'mean': scaler.mean_,
        'std_dev': scaler.scale_
    }, index=columns_to_standardize)
    
    # Save to CSV for later use
    normalization_params.to_csv('data/normalization_params.csv')
    return df     
def normalize_stream_data(df_stream):
    # Load the saved normalization parameters
    normalization_params = pd.read_csv('data/normalization_params.csv', index_col=0)
    
    # Apply normalization using the saved mean and std deviation
    for column in normalization_params.index:
        mean = normalization_params.loc[column, 'mean']
        std_dev = normalization_params.loc[column, 'std_dev']
        
        # Normalize the streaming data column
        df_stream[column] = (df_stream[column] - mean) / std_dev
    
    return df_stream


def create_lookup_table():
    # Define the lookup data
    lookup = [
            {'Column':'emp_length','Original': '< 1 year', 'Encoded': 0.5},
            {'Column':'emp_length','Original': '1 year', 'Encoded': 1},
            {'Column':'emp_length','Original': '2 years', 'Encoded': 2},
            {'Column':'emp_length','Original': '3 years', 'Encoded': 3},
            {'Column':'emp_length','Original': '4 years', 'Encoded': 4},
            {'Column':'emp_length','Original': '5 years', 'Encoded': 5},
            {'Column':'emp_length','Original': '6 years', 'Encoded': 6},
            {'Column':'emp_length','Original': '7 years', 'Encoded': 7},
            {'Column':'emp_length','Original': '8 years', 'Encoded': 8},
            {'Column':'emp_length','Original': '9 years', 'Encoded': 9},
            {'Column':'emp_length','Original': '10+ years', 'Encoded': 10},
            {'Column':'term','Original': '36 months', 'Encoded': 36},
            {'Column':'term','Original': '60 months', 'Encoded': 60},
            {'Column':'type','Original': 'INDIVIDUAL', 'Encoded': 'individual'},
            {'Column':'type','Original': 'JOINT', 'Encoded': 'joint'},
            {'Column':'type','Original': 'Joint App', 'Encoded': 'joint'},
            {'Column':'type','Original': 'DIRECT_PAY', 'Encoded': 'direct_pay'},
            {'Column':'pymnt_plan','Original': 'False', 'Encoded': 0},
            {'Column':'pymnt_plan','Original': 'True', 'Encoded': 1},
            {'Column':'annual_inc_joint','Original': '', 'Encoded': 'Value from col annual_inc'},
            {'Column':'emp_title','Original': '', 'Encoded': 'missing'},
            {'Column':'description','Original': '', 'Encoded': 'No description provided'},
            {'Column':'grade','Original': '1', 'Encoded': 'A'},
            {'Column':'grade','Original': '2', 'Encoded': 'A'},
            {'Column':'grade','Original': '3', 'Encoded': 'A'},
            {'Column':'grade','Original': '4', 'Encoded': 'A'},
            {'Column':'grade','Original': '5', 'Encoded': 'A'},
            {'Column':'grade','Original': '6', 'Encoded': 'B'},
            {'Column':'grade','Original': '7', 'Encoded': 'B'},
            {'Column':'grade','Original': '8', 'Encoded': 'B'},
            {'Column':'grade','Original': '9', 'Encoded': 'B'},
            {'Column':'grade','Original': '10', 'Encoded': 'B'},
            {'Column':'grade','Original': '11', 'Encoded': 'C'},
            {'Column':'grade','Original': '12', 'Encoded': 'C'},
            {'Column':'grade','Original': '13', 'Encoded': 'C'},
            {'Column':'grade','Original': '14', 'Encoded': 'C'},
            {'Column':'grade','Original': '15', 'Encoded': 'C'},
            {'Column':'grade','Original': '16', 'Encoded': 'D'},
            {'Column':'grade','Original': '17', 'Encoded': 'D'},
            {'Column':'grade','Original': '18', 'Encoded': 'D'},
            {'Column':'grade','Original': '19', 'Encoded': 'D'},
            {'Column':'grade','Original': '20', 'Encoded': 'D'},
            {'Column':'grade','Original': '21', 'Encoded': 'E'},
            {'Column':'grade','Original': '22', 'Encoded': 'E'},
            {'Column':'grade','Original': '23', 'Encoded': 'E'},
            {'Column':'grade','Original': '24', 'Encoded': 'E'},
            {'Column':'grade','Original': '25', 'Encoded': 'E'},
            {'Column':'grade','Original': '26', 'Encoded': 'F'},
            {'Column':'grade','Original': '27', 'Encoded': 'F'},
            {'Column':'grade','Original': '28', 'Encoded': 'F'},
            {'Column':'grade','Original': '29', 'Encoded': 'F'},
            {'Column':'grade','Original': '30', 'Encoded': 'F'},
            {'Column':'grade','Original': '31', 'Encoded': 'G'},
            {'Column':'grade','Original': '32', 'Encoded': 'G'},
            {'Column':'grade','Original': '33', 'Encoded': 'G'},
            {'Column':'grade','Original': '34', 'Encoded': 'G'},
            {'Column':'grade','Original': '35', 'Encoded': 'G'},
             {'Column':'letter_grade','Original': 'A', 'Encoded': '0'},
            {'Column':'letter_grade','Original': 'B', 'Encoded': '1'},
            {'Column':'letter_grade','Original': 'C', 'Encoded': '2'},
            {'Column':'letter_grade','Original': 'D', 'Encoded': '3'},
            {'Column':'letter_grade','Original': 'E', 'Encoded': '4'},
            {'Column':'letter_grade','Original': 'F', 'Encoded': '5'},
            {'Column':'letter_grade','Original': 'G', 'Encoded': '6'}

        ]
    
    df = pd.DataFrame(lookup, columns=['Column', 'Original', 'Encoded'])
    
    return df



def getStateName(df_org):
  df = df_org.copy()
  url = "https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971"
  payload = {}
  headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': 'AMCVS_A90F2A0D55423F537F000101%40AdobeOrg=1; s_cc=true; AMCV_A90F2A0D55423F537F000101%40AdobeOrg=-1124106680%7CMCIDTS%7C20012%7CMCMID%7C05646005586689377582520772768635170518%7CMCAAMLH-1729586519%7C6%7CMCAAMB-1729586519%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1728988920s%7CNONE%7CMCAID%7C331D9D1855191762-60000D9560649392%7CvVersion%7C5.2.0; gpv_pu=www23.statcan.gc.ca%2Fimdb%2Fp3VD.pl; gpv_pt=List%20of%20U.S.%20States%20with%20Codes%20and%20Abbreviations; gpv_pthl=blank%20theme; gpv_pc=Government%20of%20Canada%2C%20Statistics%20Canada; gpv_pqs=%3Ffunction%3Dgetvd%26tvd%3D53971; gpv_url=www23.statcan.gc.ca%2Fimdb%2Fp3VD.pl; s_ips=641; s_tp=3187; s_ppv=List%2520of%2520U.S.%2520States%2520with%2520Codes%2520and%2520Abbreviations%2C20%2C20%2C641%2C1%2C4; s_plt=6.84',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"'
  }

  response = requests.request("GET", url, headers=headers, data=payload)
  html_code = response.text   
  soup = BeautifulSoup(html_code, 'html.parser')
  state_mapping = {}
  table = soup.find('table') 
  for row in table.find_all('tr'):
      cols = row.find_all('td')  
      if len(cols) >= 3:
          state_name = cols[0].text.strip()          
          state_abbreviation = cols[2].text.strip()  
          
          state_mapping[state_abbreviation] = state_name

  df['state_name'] = df['state'].map(state_mapping)

  return df

def clean(df):
    
    df_clean=   tidy_col_names(df)
    df_clean=   choose_suitable_index(df_clean)
    df_clean2=handle_incosistent_values(df_clean)
    df_clean3=handle_missing_values(df_clean2)
    df_clean4=handle_outliers(df_clean3)
    df_clean5=add_columns(df_clean4)
    df_clean6=encode(df_clean5)
    df_clean7=normalize(df_clean6)
    # df_clean8=getStateName(df_clean7)
    lookup=create_lookup_table()
    pd.Series(df_clean7.columns).to_csv('data/one_hot_encoded_columns.csv', index=False, header=False)

    
    return df_clean7,lookup


def clean_stream(df):
    df_clean=  tidy_col_names(df)
    df_clean=  choose_suitable_index(df_clean)
    df_clean2=handle_incosistent_values(df_clean)
    df_clean3=handle_missing_values_stream(df_clean2)
    df_clean4=handle_outliers_stream(df_clean3)
    
    df_clean5 = add_month_number(df_clean4)
    df_clean5 = add_salary_can_cover(df_clean5)
    df_clean5 = calculate_installment_per_month(df_clean5)
    df_clean6=grade(df_clean5)
    
    
    df_clean7=normalize_stream_data(df_clean6)
    # df_clean8=getStateName(df_clean7)
    # df_clean7['state_name'] = df_clean7['state'].map(state_mapping)
    df_clean8 = encode_one_hot_stream(df_clean7)

    return df_clean8    




  
    