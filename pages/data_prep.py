import streamlit as st

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   initial_sidebar_state="collapsed",
                   layout="wide")
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta
from pages.chart import (revenue_by_purchase_type, 
                   purchase_type_proportion, 
                   sales_over_time, 
                   top_products_by_sales, 
                   average_order_value,
                   monthly_active_customers,
                   repeat_purchase_rate,
                   top_products_by_sales1
                   )
if st.button(label=":material/arrow_back: Back", key="back_btn", type="tertiary"):
    st.switch_page("ui.py")
    
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
### Ready to Get Started?  

**Choose how you'd like to begin:**  
1. **Use Existing Dataset:** Start exploring insights right away with our preloaded dataset, specially curated for quick analysis and demonstration.  
2. **Upload Your Own Dataset:** Bring your own data to the platform! Just make sure it follows the required format with the following columns: *InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country*  

""")

# Path to the ZIP file containing the existing data
existing_data_zip_path = "data/filtered_data.zip"

# Expected columns for the dataset
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", 
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]

data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"])
df = None

if data_option == "Use existing data":
    with zipfile.ZipFile(existing_data_zip_path, 'r') as zfile:
        # Display list of files in the ZIP
        file_list = zfile.namelist()
        csv_files = [f for f in file_list if f.endswith('.csv')]

        if not csv_files:
            st.error("No CSV file found in the ZIP.")
        else:
            # Assuming there's only one CSV file in the ZIP
            selected_file = csv_files[0]

            # Read the selected CSV file into a DataFrame
            with zfile.open(selected_file) as csvfile:
                df = pd.read_csv(csvfile)
elif data_option == "Upload new data":
    uploaded_file = st.file_uploader("Upload Dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Validation
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            st.error(f"Dataset does not match the required format. Missing columns: {missing_columns or 'None'}")
            df = None
else:
    df = None


### DATA PREP ###
if df is not None:
    st.dataframe(df, use_container_width=True)

    # Create segmented control
    selected_section = st.segmented_control(
        "Select Section",
        ["About Dataset", "Silhouette Score & Elbow Method", "K-Means Clustering"],
        default="About Dataset"
    )

    if selected_section == "About Dataset":

        ##========= EDA ==========##
        st.subheader("About Dataset", anchor=False)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            ###### Monthly Active Customers ######
            st.write("Monthly Active Customers")
            active_customers_df = monthly_active_customers(df)
            # st.line_chart(active_customers_df.set_index('InvoiceDate')['ActiveCustomers'])
            # Create Altair line chart with bullet points
            line_chart = alt.Chart(active_customers_df).mark_line().encode(
                x=alt.X('InvoiceDate:T', title='Month'),
                y=alt.Y('ActiveCustomers:Q', title='Active Customers')
            ).properties(
                width=700,
                height=400
            )

            # Add bullet points for the data points
            points = alt.Chart(active_customers_df).mark_point(filled=True, size=100).encode(
                x='InvoiceDate:T',
                y='ActiveCustomers:Q',
                tooltip=['InvoiceDate:T', 'ActiveCustomers:Q']  # Add tooltips for interactivity
            )

            # Combine the line chart and points
            final_chart = line_chart + points

            # Render in Streamlit
            st.altair_chart(final_chart, use_container_width=True)

            ##### Sales over time ####
            st.write("Sales Over Time")
            sales_df = sales_over_time(df)
            # st.line_chart(sales_df.set_index('Month')['Revenue'])
            line_chart =alt.Chart(sales_df).mark_line(color='#FDC04D').encode(
                x=alt.X('Month:T', title='Month'),
                y=alt.Y('Revenue:Q', title='Total Revenue'),
            ).properties(
                width=700,
                height=400
            )

            #Add bullet points for the data points
            points = alt.Chart(sales_df).mark_point(filled=True, size=100, color='#FDC04D').encode(
                x='Month:T',
                y='Revenue:Q',
                tooltip=['Month:T', 'Revenue:Q']  # Add tooltips for interactivity
            )
            # Combine the line chart and points
            final_chart = line_chart + points

            # Render in Streamlit
            st.altair_chart(final_chart, use_container_width=True)

        with col2:
            # Proportion of Single Item vs Multi Item Purchases
            st.write("Proportion of Single Item vs Multi Item Purchases")
            proportion_df = purchase_type_proportion(df)
            # st.bar_chart(proportion_df.set_index('Purchase Type')['Percentage'])

            # Create Altair bar chart
            bar_chart = alt.Chart(proportion_df).mark_bar().encode(
                x=alt.X('Purchase Type:O', title='Purchase Type', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Percentage:Q', title='Percentage'),
                color=alt.Color('Purchase Type:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors  scale=alt.Scale(range=['#FACD64', '#51A2F8'])
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)

            # Top products by sales volume
            st.write("Top Products by Sales Volume")
            top_products_df = top_products_by_sales(df)
            # st.write(top_products_df)

            # Create Altair bar chart
            bar_chart = alt.Chart(top_products_df).mark_bar().encode(
                x=alt.X('Description:O', title='Description', axis=alt.Axis(labelAngle=0), sort="-y"),
                y=alt.Y('Sales:Q', title='Sales'),
                color=alt.Color('Description:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)
