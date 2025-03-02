import streamlit as st
import zipfile
import pandas as pd
from pages.chart import monthly_active_customers, sales_over_time, purchase_type_proportion, top_products_by_sales

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", page_icon="📊", layout="wide")

###### Hide sidebar ######
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

if st.button(label="🔙 Back", key="back_btn", type="tertiary"):
    st.switch_page("ui.py")

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state
if 'preprocessed' not in st.session_state:
    st.session_state['preprocessed'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Preprocessing function
def preprocess_data():
    if st.session_state['df'] is not None:
        st.session_state['preprocessed'] = True
    else:
        st.error("No dataset selected! Please choose or upload a dataset first.")

# Dataset selection (only if preprocessing hasn't started)
if not st.session_state['preprocessed']:
    st.markdown("### Ready to Get Started?")

    data_option = st.selectbox("Choose Dataset", ["Select", "Use existing data", "Upload new data"], key="choose_dataset")

    if data_option == "Use existing data":
        with zipfile.ZipFile("data/filtered_data.zip", 'r') as zfile:
            file_list = [f for f in zfile.namelist() if f.endswith('.csv')]
            if file_list:
                with zfile.open(file_list[0]) as csvfile:
                    df = pd.read_csv(csvfile)
                    st.session_state['df'] = df  # Store dataframe in session state

    elif data_option == "Upload new data":
        uploaded_file = st.file_uploader("Upload Dataset", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df  # Store dataframe in session state

df = st.session_state['df']

# Display dataframe if available
if df is not None:
    st.dataframe(df, use_container_width=True)
    preprocess_button = st.button("Preprocess the data", on_click=preprocess_data)
else:
    st.warning("Please select or upload a dataset before preprocessing.")

# After preprocessing, show the new UI
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

        if df is not None:  # Ensure df is available
            with col1:
                st.write("📈 Monthly Active Customers")
                active_customers_df = monthly_active_customers(df)
                st.line_chart(active_customers_df.set_index("InvoiceDate"))

                st.write("📊 Sales Over Time")
                sales_df = sales_over_time(df)
                st.line_chart(sales_df.set_index("Month"))

            with col2:
                st.write("📌 Proportion of Single Item vs Multi Item Purchases")
                proportion_df = purchase_type_proportion(df)
                st.bar_chart(proportion_df.set_index("Purchase Type"))

                st.write("🏆 Top Products by Sales Volume")
                top_products_df = top_products_by_sales(df)
                st.bar_chart(top_products_df.set_index("Description"))
        else:
            st.error("No dataset found! Please reload the page and select a dataset.")
