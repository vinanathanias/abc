# pages/clustering.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

###### Streamlit page setup #####
st.set_page_config(
    page_title="Clustering Apps", 
    page_icon=":material/scatter_plot:", 
    initial_sidebar_state="collapsed",
    layout="wide"
)

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

# Back button to return to the preprocessing page
if st.button(label=":material/arrow_back: Back", key="back_btn", type="tertiary"):
    st.switch_page("pages/prep_visualization.py")  # Navigate back to the preprocessing page

# Main function to display clustering results
def main():
    # Retrieve the normalized data from session state
    if 'normalized_data' in st.session_state:
        normalized_data = st.session_state.normalized_data
    else:
        st.error("No normalized data found. Please go back and preprocess the data.")
        return

    # Clustering Section
    st.subheader("K-Means Clustering", anchor=False)

    # Let the user choose the number of clusters
    # st.write("Choose the Number of Clusters (k)")
    n_clusters = st.slider("Choose the Number of Clusters (k)", min_value=2, max_value=10, value=4, step=1)

    # Add a submit button to trigger clustering
    if st.button("Submit"):
        # Use a spinner to show loading while clustering is in progress
        with st.spinner("Clustering in progress..."):
            # Perform K-Means clustering with the selected k
            clustered_data = perform_kmeans_clustering(normalized_data, n_clusters)

            # Save the clustered data to session state
            st.session_state.clustered_data = clustered_data

    # Check if clustered data is available in session state
    if 'clustered_data' in st.session_state:
        clustered_data = st.session_state.clustered_data

        # Display the clustered data
        st.subheader("Clustered Data")
        st.write(clustered_data)

        # Visualize the clusters and display average scores per cluster side by side
        st.subheader("Cluster Visualization and Average Scores per Cluster")
        col1, col2 = st.columns(2)  # Create two columns

        with col1:
            # Scatter plot of recency vs monetary
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='recency', y='monetary', hue='cluster', data=clustered_data, palette='viridis', ax=ax)
            ax.set_title("Clusters: Recency vs Monetary")
            ax.set_xlabel("Recency")
            ax.set_ylabel("Monetary")
            st.pyplot(fig)

        with col2:
            # Calculate and display average scores per cluster
            st.subheader("Average Scores per Cluster")
            avg_scores_df = calculate_average_scores_per_cluster(clustered_data)
            st.dataframe(avg_scores_df)

# Function to perform K-Means clustering
def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['recency', 'frequency', 'monetary']])
    return data

# Function to calculate average scores per cluster
def calculate_average_scores_per_cluster(data):
    # Group by cluster and calculate the mean of recency, monetary, and frequency
    avg_scores_df = data.groupby('cluster').agg({
        'recency': 'mean',
        'monetary': 'mean',
        'frequency': 'mean'
    }).reset_index()
    
    # Rename columns for better readability
    avg_scores_df = avg_scores_df.rename(columns={
        'recency': 'Average Recency',
        'monetary': 'Average Monetary',
        'frequency': 'Average Frequency'
    })
    return avg_scores_df

# Run the main function
if __name__ == "__main__":
    main()
