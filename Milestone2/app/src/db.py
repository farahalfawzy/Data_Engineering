import pandas as pd
from sqlalchemy import create_engine

# Database configuration based on your Docker setup
DB_HOST = "pgdatabase"
DB_PORT = "5432"  # Port mapped to 5432 in the Docker setup
DB_NAME = "testdb"
DB_USER = "root"
DB_PASSWORD = "root"

# Connection URI for SQLAlchemy
DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize the database connection using SQLAlchemy engine
engine = create_engine(DATABASE_URI)


def save_to_database(df, table_name="fintech_data"):
    try:
        if(engine.connect()):
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Data successfully saved to table '{table_name}'.")
        else:
            print("Error: Could not establish connection to the database")
    except Exception as e:
        print(f"Error saving data to database: {e}")
        
def save_stream_data_to_database(df, table_name="fintech_data"):
    try:
        if(engine.connect()):
            df.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"Data successfully saved to table '{table_name}'.")
        else:
            print("Error: Could not establish connection to the database")
    except Exception as e:
        print(f"Error saving data to database: {e}")        

def save_lookup_table(df, table_name="lookup_data"):
    try:
        if(engine.connect()):
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Data successfully saved to table '{table_name}'.")
        else:
            print("Error: Could not establish connection to the database")
    except Exception as e:
        print(f"Error saving data to database: {e}")
        


