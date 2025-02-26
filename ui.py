import streamlit as st

###### Streamlit page setup #####
st.set_page_config(page_title="Clustering Apps", 
                   page_icon=":material/scatter_plot:", 
                   initial_sidebar_state="collapsed",
                   layout="wide")

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

if st.button("Let's get started", type="primary"):
    st.switch_page("pages/data_prep.py")
