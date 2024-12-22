import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Fintech Dashboard", layout="wide")
data = pd.read_csv( '/opt/airflow/data/fintech_transformed.csv')

# Revert log transformations for specific columns
columns_to_revert = ['annual_inc', 'annual_inc_joint']
data[columns_to_revert] = np.expm1(data[columns_to_revert])

st.title("Fintech Dashboard")
st.write("Created by Farah Maher Alfawzy - ID: 52_0324")

# Organize layout into tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Loan Amounts by Grade", "Loan vs Income by State", "Issuance Trend by Year", "Avg Loan by State", "Grade Distribution"
])

#    1. What is the distribution of loan amounts across different grades? (use letter
# grades or encoded grades (1-7) not grades from the uncleaned file)
# Hint: Use a box plot or violin plot to show the spread of loan amounts
# for each grade.
with tab1:
    st.header("Distribution of Loan Amounts by Letter Grade")
    # Sort data by letter grade
    fig1 = px.box(data.sort_values('letter_grade'), x='letter_grade', y='loan_amount', 
                title='Loan Amounts by Letter Grade', color_discrete_sequence=["#FF8C00"])
    st.plotly_chart(fig1, use_container_width=True)
    

#     2. How does the loan amount relate to annual income across states ? (Interactive)
# Hint: Use a scatter plot with loan amount and annual income (original
# values).
# Add color-coding based on loan status (e.g., fully paid, default).
# Have a dropdown with all states to filter either each state (unique
# value of states) or all which shows all states (Hint: you need to check
# if the value is 'all' don't apply a filter), make all the default and
# first option.
with tab2:
    st.header("Loan Amount vs. Annual Income by State")
    selected_state = st.selectbox("Select State", ["All"] + list(data["state"].unique()))
    filtered_data = data if selected_state == "All" else data[data["state"] == selected_state]
    fig2 = px.scatter(filtered_data, x="annual_inc", y="loan_amount", 
                        color="loan_status", 
                        title="Loan Amount vs. Annual Income")
    st.plotly_chart(fig2, use_container_width=True)

#What is the trend of loan issuance over the months (number of loans per month),
# filtered by year? (Interactive)
# Hint: Use a line graph showing the count or total loan amount issued per month.
# Filter using a dropdown with the years available.
with tab3:
    st.header("Total Loan Issuance Trend by Year")
    total_yearly_data = data.groupby([data["issue_date"].str[:4], "month_number"])['loan_amount'].sum().reset_index()
    selected_year = st.selectbox("Filter by Year", ["All"] + sorted(data["issue_date"].str[:4].unique()))
    if selected_year == "All":
        fig3 = px.line(total_yearly_data, x="month_number", y="loan_amount", 
                        color="issue_date", 
                        title="Total Loan Issuance Trend by Year", 
                        labels={"month_number": "Month", "loan_amount": "Total Loan Amount", "issue_date": "Year"})
    else:
        yearly_total_data = total_yearly_data[total_yearly_data["issue_date"] == selected_year]
        fig3 = px.line(yearly_total_data, x="month_number", y="loan_amount", 
                        title=f"Total Loan Issuance Trend in {selected_year}", 
                        labels={"month_number": "Month", "loan_amount": "Total Loan Amount"})
    st.plotly_chart(fig3, use_container_width=True)

# 4. Which states have the highest average loan amount?
# Hint: Use a bar chart to display average loan amounts for each state, or
# an interactive choropleth map to enhance the visualization (optional),
# showing states shaded by their average loan amounts.
with tab4:
    st.header("States with Highest Average Loan Amount")
    avg_loans = data.groupby("state")["loan_amount"].mean().reset_index().sort_values("loan_amount", ascending=False)
    fig4 = px.choropleth(avg_loans, locations="state", locationmode="USA-states", color="loan_amount", 
                            color_continuous_scale=px.colors.sequential.Peach, scope="usa", 
                            title="Average Loan Amount by State")
    st.plotly_chart(fig4, use_container_width=True)

# 5. What is the percentage distribution of loan grades in the dataset?
# Hint: Use a histogram or kde plot.
with tab5:
    st.header("Percentage Distribution of Loan Grades")
    fig5 = px.histogram(data.sort_values('letter_grade'), x="letter_grade", nbins=7, 
                        title="Percentage Distribution of Loan Grades", histnorm="percent",color_discrete_sequence=["#FFA07A"])
    st.plotly_chart(fig5, use_container_width=True)
