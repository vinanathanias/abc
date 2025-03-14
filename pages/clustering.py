# pages/clustering.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px  # For parallel coordinates plot

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

# Function to calculate the transition matrix
def calculate_transition_matrix(clustered_data):
    # Ensure the data has a 'month_year_end' column for time-based transitions
    if 'month_year_end' not in clustered_data.columns:
        st.error("The dataset does not contain a 'month_year_end' column for time-based transitions.")
        return None

    # Sort the data by CustomerID and month_year_end
    clustered_data = clustered_data.sort_values(by=['CustomerID', 'month_year_end'])

    # Create a transition matrix
    unique_clusters = sorted(clustered_data['cluster'].unique())
    transition_matrix = pd.DataFrame(
        np.zeros((len(unique_clusters), len(unique_clusters))),
        index=unique_clusters,
        columns=unique_clusters
    )

    # Calculate transitions
    for customer_id, customer_data in clustered_data.groupby('CustomerID'):
        customer_data = customer_data.sort_values(by='month_year_end')
        previous_cluster = None
        for _, row in customer_data.iterrows():
            current_cluster = row['cluster']
            if previous_cluster is not None:
                transition_matrix.loc[previous_cluster, current_cluster] += 1
            previous_cluster = current_cluster

    # Normalize the transition matrix to get probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    return transition_matrix

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

# Function to create a parallel coordinates plot
def parallel_coordinates_plot(data):
    # Ensure the data has the required columns
    if not all(col in data.columns for col in ['recency', 'frequency', 'monetary', 'cluster']):
        st.error("The dataset does not contain the required columns for the parallel coordinates plot.")
        return

    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        data,
        color="cluster",  # Use cluster for coloring
        dimensions=['recency', 'frequency', 'monetary'],  # Dimensions to plot
        labels={
            'recency': 'Recency',
            'frequency': 'Frequency',
            'monetary': 'Monetary',
            'cluster': 'Cluster'
        },
        color_continuous_scale=px.colors.sequential.Viridis  # Color scale
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to handle clustering when the submit button is clicked
def handle_clustering():
    # Retrieve the normalized data from session state
    if 'normalized_data' in st.session_state:
        normalized_data = st.session_state.normalized_data
    else:
        st.error("No normalized data found. Please go back and preprocess the data.")
        return

    # Perform K-Means clustering with the selected k
    n_clusters = st.session_state.n_clusters
    clustered_data = perform_kmeans_clustering(normalized_data, n_clusters)

    # Save the clustered data to session state
    st.session_state.clustered_data = clustered_data

# Function to handle quitting or finishing the process
def handle_quit():
    # Clear the session state
    st.session_state.clear()
    # Navigate back to the home page
    st.switch_page("main.py")

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
    n_clusters = st.slider("Choose the Number of Clusters (k)", min_value=2, max_value=10, value=4, step=1)

    # Save the selected number of clusters to session state
    st.session_state.n_clusters = n_clusters

    # Add a submit button to trigger clustering
    if st.button("Submit", on_click=handle_clustering):
        # Use a spinner to show loading while clustering is in progress
        with st.spinner("Clustering in progress..."):
            time.sleep(5)  # Simulate a delay for demonstration purposes

    # Check if clustered data is available in session state
    if 'clustered_data' in st.session_state:
        clustered_data = st.session_state.clustered_data

        # Display the clustered data
        st.subheader("Clustering Result")
        st.write("Here is the clustered data after applying K-Means clustering:")
        st.dataframe(clustered_data, use_container_width=True)

        # Visualize the clusters and display average scores per cluster
        st.subheader("Cluster Visualization and Average Scores per Cluster")

        # Create three columns for the scatter plots
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Recency vs Monetary**")
            st.write("This scatter plot shows the relationship between Recency and Monetary values for each cluster.")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='recency', y='monetary', hue='cluster', data=clustered_data, palette='viridis', ax=ax)
            ax.set_title("Recency vs Monetary")
            ax.set_xlabel("Recency")
            ax.set_ylabel("Monetary")
            st.pyplot(fig)

        with col2:
            st.write("**Recency vs Frequency**")
            st.write("This scatter plot shows the relationship between Recency and Frequency values for each cluster.")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='recency', y='frequency', hue='cluster', data=clustered_data, palette='viridis', ax=ax)
            ax.set_title("Recency vs Frequency")
            ax.set_xlabel("Recency")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col3:
            st.write("**Frequency vs Monetary**")
            st.write("This scatter plot shows the relationship between Frequency and Monetary values for each cluster.")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='frequency', y='monetary', hue='cluster', data=clustered_data, palette='viridis', ax=ax)
            ax.set_title("Frequency vs Monetary")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Monetary")
            st.pyplot(fig)

        # Display average scores per cluster in a separate section
        st.subheader("Average Scores per Cluster")
        st.write("The table below shows the average Recency, Frequency, and Monetary values for each cluster.")
        avg_scores_df = calculate_average_scores_per_cluster(clustered_data)
        st.dataframe(avg_scores_df, use_container_width=True)

        # Calculate and display the transition matrix
        st.subheader("Markov Chains Transition Matrix")
        st.write("The transition matrix shows the probability of moving from one cluster to another over time.")
        transition_matrix = calculate_transition_matrix(clustered_data)

        coll0, coll1, coll2 = st.columns([0.1, 2, 2])
        with coll1:
            if transition_matrix is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    transition_matrix,
                    annot=True,  # Show annotations
                    cmap="YlGnBu",  # Color map
                    fmt=".4f",  # Format annotations to 4 decimal places
                    ax=ax
                )
                ax.set_title("Transition Matrix (Cluster to Cluster)")
                ax.set_xlabel("Next Cluster")
                ax.set_ylabel("Current Cluster")
                st.pyplot(fig)

        # Add Parallel Coordinates Plot
        st.subheader("Parallel Coordinates Plot")
        st.write("This plot visualizes clusters across multiple dimensions (Recency, Frequency, Monetary).")
        parallel_coordinates_plot(clustered_data)

        # Add a Quit/Done button
        if st.button("Quit/Done", type="primary"):
            handle_quit()

# Run the main function
if __name__ == "__main__":
    main()
