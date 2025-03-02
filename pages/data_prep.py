import streamlit as st

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   initial_sidebar_state="collapsed",
                   layout="wide")

import zipfile
import pandas as pd
import numpy as np
from pages.chart import (
    revenue_by_purchase_type, purchase_type_proportion, sales_over_time, 
    top_products_by_sales, monthly_active_customers
)

###### Hide sidebar ######
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

if st.button(label=":material/arrow_back: Back", key="back_btn", type="tertiary"):
    st.switch_page("ui.py")

st.markdown("<br>", unsafe_allow_html=True)

# Ensure session state is initialized properly
if 'preprocessed' not in st.session_state:
    st.session_state['preprocessed'] = False

# Preprocessing function
def preprocess_data():
    st.session_state['preprocessed'] = True  # Set flag to True when preprocessing is done

# Only show dataset selection if preprocessing hasn't started
if not st.session_state['preprocessed']:
    st.markdown("""
    ### Ready to Get Started?  

    **Choose how you'd like to begin:**  
    1. **Use Existing Dataset:** Start exploring insights right away.  
    2. **Upload Your Own Dataset:** Bring your own data!  

    """)

    # Dataset selection
    data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"], key="choose_dataset")
    df = None

    if data_option == "Use existing data":
        with zipfile.ZipFile("data/filtered_data.zip", 'r') as zfile:
            file_list = [f for f in zfile.namelist() if f.endswith('.csv')]
            if file_list:
                with zfile.open(file_list[0]) as csvfile:
                    df = pd.read_csv(csvfile)
    elif data_option == "Upload new data":
        uploaded_file = st.file_uploader("Upload Dataset", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    # Store dataframe in session state
    st.session_state['df'] = df

# Load dataset from session state
df = st.session_state.get('df', None)

if df is not None:
    st.dataframe(df, use_container_width=True)
    if st.button("Preprocess the data"):
        preprocess_data()

# After preprocessing, display new UI elements
if st.session_state['preprocessed']:
    st.markdown("### Data Preprocessed Successfully!")
    selected_section = st.segmented_control(
        "Select Section",
        ["About Dataset", "Silhouette Score & Elbow Method", "K-Means Clustering"],
        default="About Dataset"
    )

    if selected_section == "About Dataset":
        st.subheader("About Dataset", anchor=False)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.write("Monthly Active Customers")
            active_customers_df = monthly_active_customers(df)
            st.line_chart(active_customers_df.set_index("InvoiceDate"))

            st.write("Sales Over Time")
            sales_df = sales_over_time(df)
            st.line_chart(sales_df.set_index("Month"))

        with col2:
            st.write("Proportion of Single Item vs Multi Item Purchases")
            proportion_df = purchase_type_proportion(df)
            st.bar_chart(proportion_df.set_index("Purchase Type"))

            st.write("Top Products by Sales Volume")
            top_products_df = top_products_by_sales(df)
            st.bar_chart(top_products_df.set_index("Description"))
