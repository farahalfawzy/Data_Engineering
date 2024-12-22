from functions import extract_clean, transform, load_to_db
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


print('DAG is working')
# Define the DAG
default_args = {
    "owner": "Farah Maher Alfawzy",
    "depends_on_past": False,
    'start_date': days_ago(0),
    "retries": 1,
}

dag = DAG(
    'fintech_pipeline',
    default_args=default_args,
    description='fintch pipeline',
)

with DAG(
    dag_id = 'fintech_pipeline',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['fintech-pipeline'],
)as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs={
            'filename': '/opt/airflow/data/fintech_data_31_52_0324.csv'
        }
        
    )

    transform = PythonOperator(
        task_id = 'transform',
        python_callable = transform,
        op_kwargs={
            'filename': '/opt/airflow/data/fintech_clean.csv'
        }
       
    )

    load_to_db = PythonOperator(
        task_id = 'load_to_db',
        python_callable = load_to_db,
        op_kwargs={
            'filename': '/opt/airflow/data/fintech_transformed.csv'
        }
        
    )
    
    run_dashboard = BashOperator(
        task_id = 'run_dashboard',
        bash_command = 'streamlit run /opt/airflow/dags/fintech_dashboard.py --server.port 8501  --server.address 0.0.0.0  --server.headless true',
        dag = dag
    )
   
    extract_clean_task >> transform >> load_to_db >> run_dashboard