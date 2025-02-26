import streamlit as st
import zipfile
import pandas as pd

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   layout="wide")

### LAYOUT ###
st.header("Welcome to Clustering Apps", anchor=False)
with st.container(border=True):
    st.write("This is inside the container")

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

# Display the DataFrame if it exists
if df is not None:
    st.dataframe(df)
