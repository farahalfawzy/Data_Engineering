from cleaning import clean,clean_stream
from db import save_to_database, save_lookup_table,save_stream_data_to_database

import os
import pandas as pd
from run_producer import start_producer, stop_container
from consumer import start_consumer, get_next_message, close_consumer
import time
column_names = [
        "Customer Id", "Emp Title", "Emp Length", "Home Ownership", "Annual Inc", 
        "Annual Inc Joint", "Verification Status", "Zip Code", "Addr State", 
        "Avg Cur Bal", "Tot Cur Bal", "Loan Id", "Loan Status", "Loan Amount", 
        "State", "Funded Amount", "Term", "Int Rate", "Grade", "Issue Date", 
        "Pymnt Plan", "Type", "Purpose", "Description"
    ]

def load_raw_data(file_path):
    """Loads raw data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Raw data loaded successfully.")
        return df
    except FileNotFoundError:
        print("Raw data file not found.")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found

def main():
    raw_data_path = 'data/fintech_data_31_52_0324.csv'
    cleaned_data_path = 'data/fintech_data_MET_P2_52_0324_clean.csv'

    
    if os.path.exists(cleaned_data_path):
        print("Cleaned data already exists. Loading from file...")
        cleaned_data = pd.read_csv(cleaned_data_path)
    else:
        print("No cleaned data found. Starting the cleaning process...")
        raw_data = load_raw_data(raw_data_path)
        if raw_data.empty:
            print("No raw data to process.")
            return  # Exit if there is no raw data
        cleaned_data,lookup = clean(raw_data)
        cleaned_data.to_csv(cleaned_data_path, index=False)
        lookup.to_csv('data/lookup_fintech_data_MET_P2_52_0324.csv', index=False)
        print("Cleaned data saved to CSV.")
        print("Saving cleaned data to database...")
        save_to_database(cleaned_data)
        save_lookup_table(lookup)
    print("Data saved successfully.")
    time.sleep(30)
    id = start_producer()
    consumer = start_consumer()  
    i=0
    while True:
        message_value = get_next_message(consumer)
        if not message_value is None:
            if   message_value =="EOF":
                print("End of stream detected. Closing consumer.")
                break
           
            new_message = pd.DataFrame([message_value], columns=column_names)
            cleaned_new_data = clean_stream(new_message)
            cleaned_new_data.to_csv(cleaned_data_path, mode='a', index=False, header=False)
            save_stream_data_to_database(cleaned_new_data)
            message_value=None
        
        
    close_consumer(consumer)
    print("Consumer closed successfully.")
    stop_container(id)
   
    
        
        
        
        

if __name__ == "__main__":
    main()
