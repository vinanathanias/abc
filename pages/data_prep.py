# pages/page1.py

import streamlit as st
import zipfile
import pandas as pd

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   initial_sidebar_state="collapsed",
                   layout="wide")

###### Hide sidebar ######
st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none
            }

            [data-testid="collapsedControl"] {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)

if st.button(label=":material/arrow_back: Back", key="back_btn", type="tertiary"):
    st.switch_page("main.py")  # Navigate back to the main page
    
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

data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"], key="choose_dataset")
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

# Check if dataframe is available
if df is not None:
    # Display the dataframe
    st.dataframe(df, use_container_width=True)

    # Add a button to navigate to the preprocessing and visualization page
    if st.button("Preprocess the data"):
        # Pass the dataframe to the next page using query parameters
        st.switch_page("pages/prep_visualization.py")


# import streamlit as st

# ###### Streamlit page setup #####
# st.set_page_config(page_title="Clustering Apps", 
#                    page_icon=":material/scatter_plot:", 
#                    initial_sidebar_state="collapsed",
#                    layout="wide")
# import zipfile
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import silhouette_scorae
# from sklearn.cluster import KMeans
# import altair as alt
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from datetime import timedelta
# from pages.chart import (revenue_by_purchase_type, 
#                    purchase_type_proportion, 
#                    sales_over_time, 
#                    top_products_by_sales, 
#                    average_order_value,
#                    monthly_active_customers,
#                    repeat_purchase_rate,
#                    top_products_by_sales1
#                    )
# ###### Hide sidebar ######
# st.markdown("""
#             <style>
#             [data-testid="stSidebar"] {
#                 display: none
#             }

#             [data-testid="collapsedControl"] {
#                 display: none
#             }
#             </style>
#             """, unsafe_allow_html=True)

# if st.button(label=":material/arrow_back: Back", key="back_btn", type="tertiary"):
#     st.switch_page("ui.py")
    
# st.markdown("<br>", unsafe_allow_html=True)

# st.markdown("""
# ### Ready to Get Started?  

# **Choose how you'd like to begin:**  
# 1. **Use Existing Dataset:** Start exploring insights right away with our preloaded dataset, specially curated for quick analysis and demonstration.  
# 2. **Upload Your Own Dataset:** Bring your own data to the platform! Just make sure it follows the required format with the following columns: *InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country*  

# """)

# # Path to the ZIP file containing the existing data
# existing_data_zip_path = "data/filtered_data.zip"

# # Expected columns for the dataset
# EXPECTED_COLUMNS = [
#     "InvoiceNo", "StockCode", "Description", "Quantity", 
#     "InvoiceDate", "UnitPrice", "CustomerID", "Country"
# ]

# data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"], key="choose_dataset")
# df = None

# if data_option == "Use existing data":
#     with zipfile.ZipFile(existing_data_zip_path, 'r') as zfile:
#         # Display list of files in the ZIP
#         file_list = zfile.namelist()
#         csv_files = [f for f in file_list if f.endswith('.csv')]

#         if not csv_files:
#             st.error("No CSV file found in the ZIP.")
#         else:
#             # Assuming there's only one CSV file in the ZIP
#             selected_file = csv_files[0]

#             # Read the selected CSV file into a DataFrame
#             with zfile.open(selected_file) as csvfile:
#                 df = pd.read_csv(csvfile)
# elif data_option == "Upload new data":
#     uploaded_file = st.file_uploader("Upload Dataset", type="csv")
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)

#         # Validation
#         missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
#         if missing_columns:
#             st.error(f"Dataset does not match the required format. Missing columns: {missing_columns or 'None'}")
#             df = None
# else:
#     df = None

# # Initialize session state for preprocessing if not already set
# if 'preprocessed' not in st.session_state:
#     st.session_state['preprocessed'] = False

# # Function to preprocess data
# def preprocess_data():
#     # Set a flag in session state to indicate preprocessing is done
#     st.session_state['preprocessed'] = True

# # Function to display visualizations
# def display_visualizations():
#     # Display the segmented control only after preprocessing
#     selected_section = st.segmented_control(
#         "Select Section",
#         ["About Dataset", "Silhouette Score & Elbow Method", "K-Means Clustering"],
#         default="About Dataset"
#     )

#     if selected_section == "About Dataset":
#         ##========= EDA ==========##
#         st.subheader("About Dataset", anchor=False)
#         col1, col2 = st.columns(2, gap="large")

#         with col1:
#             ###### Monthly Active Customers ######
#             st.write("Monthly Active Customers")
#             active_customers_df = monthly_active_customers(df)
            
#             # Create Altair line chart with bullet points
#             line_chart = alt.Chart(active_customers_df).mark_line().encode(
#                 x=alt.X('InvoiceDate:T', title='Month'),
#                 y=alt.Y('ActiveCustomers:Q', title='Active Customers')
#             ).properties(
#                 width=700,
#                 height=400
#             )

#             # Add bullet points for the data points
#             points = alt.Chart(active_customers_df).mark_point(filled=True, size=100).encode(
#                 x='InvoiceDate:T',
#                 y='ActiveCustomers:Q',
#                 tooltip=['InvoiceDate:T', 'ActiveCustomers:Q']  # Add tooltips for interactivity
#             )

#             # Combine the line chart and points
#             final_chart = line_chart + points

#             # Render in Streamlit
#             st.altair_chart(final_chart, use_container_width=True)

#             ##### Sales over time ####
#             st.write("Sales Over Time")
#             sales_df = sales_over_time(df)
            
#             line_chart = alt.Chart(sales_df).mark_line(color='#FDC04D').encode(
#                 x=alt.X('Month:T', title='Month'),
#                 y=alt.Y('Revenue:Q', title='Total Revenue'),
#             ).properties(
#                 width=700,
#                 height=400
#             )

#             # Add bullet points for the data points
#             points = alt.Chart(sales_df).mark_point(filled=True, size=100, color='#FDC04D').encode(
#                 x='Month:T',
#                 y='Revenue:Q',
#                 tooltip=['Month:T', 'Revenue:Q']  # Add tooltips for interactivity
#             )
            
#             # Combine the line chart and points
#             final_chart = line_chart + points

#             # Render in Streamlit
#             st.altair_chart(final_chart, use_container_width=True)

#         with col2:
#             # Proportion of Single Item vs Multi Item Purchases
#             st.write("Proportion of Single Item vs Multi Item Purchases")
#             proportion_df = purchase_type_proportion(df)

#             # Create Altair bar chart
#             bar_chart = alt.Chart(proportion_df).mark_bar().encode(
#                 x=alt.X('Purchase Type:O', title='Purchase Type', axis=alt.Axis(labelAngle=0)),
#                 y=alt.Y('Percentage:Q', title='Percentage'),
#                 color=alt.Color('Purchase Type:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors
#             ).properties(
#                 width=600,
#                 height=400
#             )

#             st.altair_chart(bar_chart, use_container_width=True)

#             # Top products by sales volume
#             st.write("Top Products by Sales Volume")
#             top_products_df = top_products_by_sales(df)

#             # Create Altair bar chart
#             bar_chart = alt.Chart(top_products_df).mark_bar().encode(
#                 x=alt.X('Description:O', title='Description', axis=alt.Axis(labelAngle=0), sort="-y"),
#                 y=alt.Y('Sales:Q', title='Sales'),
#                 color=alt.Color('Description:N', scale=alt.Scale(scheme='viridis'))  # Custom bar colors
#             ).properties(
#                 width=600,
#                 height=400
#             )

#             st.altair_chart(bar_chart, use_container_width=True)

# ### DATA PREP ###

# # Add this function to handle outliers
# def handle_outliers(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     # Cap the outliers
#     data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
#     return data

# # Add this function to normalize data using MinMaxScaler
# def normalize_data(data, columns):
#     scaler = MinMaxScaler()
#     data[columns] = scaler.fit_transform(data[columns])
#     return data

# # Function to calculate RFM data
# def calculate_rfm(df):
#     # Convert InvoiceDate to datetime
#     df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
#     df['month_end'] = df['InvoiceDate'] + pd.offsets.MonthEnd(0)
#     df['month_year_end'] = df['month_end'].dt.strftime('%Y-%m')
#     df['recency'] = (df['month_end'] - df['InvoiceDate']).dt.days
#     df['monetary'] = df['UnitPrice'] * df['Quantity']
#     df['frequency'] = df.groupby(['CustomerID', 'month_end'])['InvoiceNo'].transform('count')
    
#     return df.groupby(['CustomerID', 'month_year_end']).agg(
#         recency=('recency', 'mean'),
#         monetary=('monetary', 'mean'),
#         frequency=('frequency', 'sum')
#     ).reset_index()

# # Function to visualize RFM data
# def visualize_rfm_data(data, title):
#     # Visualize outliers using scatter plots
#     fig, ax = plt.subplots(1, 3, figsize=(18, 6))

#     sns.scatterplot(x='recency', y='monetary', data=data, ax=ax[0])
#     ax[0].set_title(f"{title}: Recency vs Monetary")
#     ax[0].set_xlabel("Recency")
#     ax[0].set_ylabel("Monetary")

#     sns.scatterplot(x='frequency', y='monetary', data=data, ax=ax[1])
#     ax[1].set_title(f"{title}: Frequency vs Monetary")
#     ax[1].set_xlabel("Frequency")
#     ax[1].set_ylabel("Monetary")

#     sns.scatterplot(x='recency', y='frequency', data=data, ax=ax[2])
#     ax[2].set_title(f"{title}: Recency vs Frequency")
#     ax[2].set_xlabel("Recency")
#     ax[2].set_ylabel("Frequency")

#     st.pyplot(fig)
  
# # Modify the existing code to include outlier handling and normalization
# if df is not None:
#     # Display the dataframe
#     st.dataframe(df, use_container_width=True)

#     # Display Preprocess button with on_click parameter
#     if st.button("Preprocess the data", on_click=preprocess_data):
#         # Calculate RFM data
#         monthly_data = calculate_rfm(df)

#         # Visualize data BEFORE handling outliers
#         st.header("Data Visualization BEFORE Handling Outliers", anchor=False)
#         visualize_rfm_data(monthly_data, "Before Handling Outliers")

#         # Handle outliers in the RFM metrics
#         monthly_data = handle_outliers(monthly_data, 'recency')
#         monthly_data = handle_outliers(monthly_data, 'frequency')
#         monthly_data = handle_outliers(monthly_data, 'monetary')

#         # Visualize data AFTER handling outliers
#         st.header("Data Visualization AFTER Handling Outliers", anchor=False)
#         visualize_rfm_data(monthly_data, "After Handling Outliers")

#         # Normalize the RFM metrics
#         monthly_data = normalize_data(monthly_data, ['recency', 'frequency', 'monetary'])
#         st.header("Data AFTER normalized", anchor=False)
#         st.write(monthly_data)

#         # Set a flag in session state to indicate preprocessing is done
#         st.session_state['preprocessed'] = True

#     # Display visualizations only if data is preprocessed
#     if st.session_state.get('preprocessed'):
#         display_visualizations()






# # # Check if dataframe is available
# # if df is not None:
# #     # Display the dataframe
# #     st.dataframe(df, use_container_width=True)

# #     # Display Preprocess button with on_click parameter
# #     st.button("Preprocess the data", on_click=preprocess_data)

# #     # Display visualizations only if data is preprocessed
# #     if st.session_state.get('preprocessed'):
# #         display_visualizations()
      
# #         ### RFM (Monthly Only) ###
# #         def calculate_rfm(df):
# #             # Convert InvoiceDate to datetime
# #             df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
# #             df['month_end'] = df['InvoiceDate'] + pd.offsets.MonthEnd(0)
# #             df['month_year_end'] = df['month_end'].dt.strftime('%Y-%m')
# #             df['recency'] = (df['month_end'] - df['InvoiceDate']).dt.days
# #             df['monetary'] = df['UnitPrice'] * df['Quantity']
# #             df['frequency'] = df.groupby(['CustomerID', 'month_end'])['InvoiceNo'].transform('count')
    
# #             return df.groupby(['CustomerID', 'month_year_end']).agg(
# #                 recency=('recency', 'mean'),
# #                 monetary=('monetary', 'mean'),
# #                 frequency=('frequency', 'sum')
# #             ).reset_index()
    
# #         # Monthly RFM Data
# #         monthly_data = calculate_rfm(df)
# #         st.header("RFM data per customer ID per months", anchor=False)
# #         st.dataframe(monthly_data, use_container_width=True)

# #         ### Outlier Detection and Visualization ###
# #         st.header("Outlier Detection in RFM Metrics", anchor=False)

# #         # Function to detect outliers using IQR
# #         def detect_outliers_iqr(data, column):
# #             Q1 = data[column].quantile(0.25)
# #             # st.write("Q1", Q1)
# #             Q3 = data[column].quantile(0.75)
# #             # st.write("Q3", Q3)
# #             IQR = Q3 - Q1
# #             # st.write("IQR", IQR)
# #             lower_bound = Q1 - 1.5 * IQR
# #             upper_bound = Q3 + 1.5 * IQR
# #             # st.write("lower_bound", lower_bound)
# #             # st.write("upper_bound", upper_bound)
# #             outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
# #             # st.write("outliers", outliers)
# #             return outliers

# #         # Detect outliers for each RFM metric
# #         recency_outliers = detect_outliers_iqr(monthly_data, 'recency')
# #         frequency_outliers = detect_outliers_iqr(monthly_data, 'frequency')
# #         monetary_outliers = detect_outliers_iqr(monthly_data, 'monetary')

        
# #         # st.write("recency_outliers", recency_outliers)
# #         # st.write("frequency_outliers", frequency_outliers)
# #         # st.write("monetary_outliers", monetary_outliers)

# #         # Visualize outliers using scatter plots
# #         fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# #         sns.scatterplot(x='recency', y='monetary', data=monthly_data, ax=ax[0])
# #         ax[0].set_title("Recency vs Monetary")
# #         ax[0].set_xlabel("Recency")
# #         ax[0].set_ylabel("Monetary")

# #         sns.scatterplot(x='frequency', y='monetary', data=monthly_data, ax=ax[1])
# #         ax[1].set_title("Frequency vs Monetary")
# #         ax[1].set_xlabel("Frequency")
# #         ax[1].set_ylabel("Monetary")

# #         sns.scatterplot(x='recency', y='frequency', data=monthly_data, ax=ax[2])
# #         ax[2].set_title("Recency vs Frequency")
# #         ax[2].set_xlabel("Recency")
# #         ax[2].set_ylabel("Frequency")

# #         st.pyplot(fig)
