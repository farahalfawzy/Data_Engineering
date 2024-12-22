# Data_Engineering


This repository contains the implementation of a Data Engineering project focusing on fintech datasets. The project is divided into four milestones, each building upon the previous one to achieve end-to-end data processing and analysis. Below are the details of the milestones and the functionality implemented.

## Table of Contents
- [Introduction](#introduction)
- [Milestones Overview](#milestones-overview)
  - [Milestone 1](#milestone-1)
  - [Milestone 2](#milestone-2)
  - [Milestone 3](#milestone-3)
  - [Milestone 4](#milestone-4)
- [Technologies Used](#technologies-used)


## Introduction

The project demonstrates essential data engineering concepts, including data cleaning, transformation, feature engineering, stream processing, and visualization. The main objective is to handle a fintech dataset through various stages, culminating in a dashboard visualization for insights.

## Milestones Overview

### Milestone 1
- **Objective:** Perform Exploratory Data Analysis (EDA), clean the dataset, and engineer new features.
- **Key Tasks:**
  - Conduct EDA and generate visualizations.
  - Clean data by handling missing values, duplicates, and outliers.
  - Engineer features like monthly installments, letter grades, and loan coverage.
  - Save the cleaned dataset and a lookup table programmatically.


### Milestone 2
- **Objective:** Package the project in a Docker container and implement a Kafka-based data stream.
- **Key Tasks:**
  - Package the data cleaning code into functions and integrate it with Docker.
  - Use Kafka for streaming data and save processed streams into a PostgreSQL database.
  - Utilize a Docker Compose setup including Postgres, Kafka, and PgAdmin.


### Milestone 3
- **Objective:** Perform advanced data cleaning and feature engineering using PySpark.
- **Key Tasks:**
  - Load, clean, and analyze data using PySpark.
  - Engineer features based on historical loan data.
  - Compare SQL vs PySpark for key analyses.
  - Save results to a Postgres database.


### Milestone 4
- **Objective:** Develop an ETL pipeline and create an interactive dashboard.
- **Key Tasks:**
  - Implement an ETL pipeline using Apache Airflow.
  - Build a dashboard using Streamlit for insights.
  - Integrate the dashboard with the ETL pipeline.

## Technologies Used
- **Programming Languages:** Python, SQL
- **Tools and Libraries:** 
  - Data Processing: PySpark, Pandas
  - Data Streaming: Kafka, kafka-python
  - ETL Orchestration: Apache Airflow
  - Database: PostgreSQL
  - Visualization: Plotly Dash, Streamlit
- **Containerization:** Docker, Docker Compose

