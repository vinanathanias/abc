import streamlit as st

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   layout="wide")

### LAYOUT ###
st.header("Welcome to Clustering Apps", anchor=False, divider="grey")

with st.container():
    st.write("""  
    Harness the power of **K-Means clustering** to gain valuable insights into your online retail dataset. 
    Our platform allows you to effortlessly segment customers, helping you understand their behavior and tailor your marketing strategies.  
    
    **Key Features:**  
    - **Flexible Clustering:** Adjust the number of clusters to suit your needs, or leverage the recommended optimum clusters using Silhouette Score and Elbow Score for the best segmentation results.  
    - **Data Visualization:** Easily identify customer patterns and behaviors through intuitive visualizations, empowering you to make informed business decisions.  
    - **Simple Data Upload:** Our tool accepts datasets with the following columns: *InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country* 
    
    Start exploring your customer data today and uncover meaningful insights to grow your business!
    """)

# st.page_link("data_prep.py", label="Let's get started", icon="✅", disabled=True)
