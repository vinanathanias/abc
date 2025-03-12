# pages/prep_visualization.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pages.chart import (
    revenue_by_purchase_type, 
    purchase_type_proportion, 
    sales_over_time, 
    top_products_by_sales, 
    average_order_value,
    monthly_active_customers,
    repeat_purchase_rate,
    top_products_by_sales1
)

# Function to handle outliers
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap the outliers
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

# Function to normalize data using MinMaxScaler
def normalize_data(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Function to calculate RFM data
def calculate_rfm(df):
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
    df['month_end'] = df['InvoiceDate'] + pd.offsets.MonthEnd(0)
    df['month_year_end'] = df['month_end'].dt.strftime('%Y-%m')
    df['recency'] = (df['month_end'] - df['InvoiceDate']).dt.days
    df['monetary'] = df['UnitPrice'] * df['Quantity']
    df['frequency'] = df.groupby(['CustomerID', 'month_end'])['InvoiceNo'].transform('count')
    
    return df.groupby(['CustomerID', 'month_year_end']).agg(
        recency=('recency', 'mean'),
        monetary=('monetary', 'mean'),
        frequency=('frequency', 'sum')
    ).reset_index()

# Function to visualize RFM data
def visualize_rfm_data(data, title):
    # Visualize outliers using scatter plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    sns.scatterplot(x='recency', y='monetary', data=data, ax=ax[0])
    ax[0].set_title(f"{title}: Recency vs Monetary")
    ax[0].set_xlabel("Recency")
    ax[0].set_ylabel("Monetary")

    sns.scatterplot(x='frequency', y='monetary', data=data, ax=ax[1])
    ax[1].set_title(f"{title}: Frequency vs Monetary")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Monetary")

    sns.scatterplot(x='recency', y='frequency', data=data, ax=ax[2])
    ax[2].set_title(f"{title}: Recency vs Frequency")
    ax[2].set_xlabel("Recency")
    ax[2].set_ylabel("Frequency")

    st.pyplot(fig)

# Function to perform K-Means clustering
def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['recency', 'frequency', 'monetary']])
    return data

# Function to plot the Elbow Method
def plot_elbow_method(data):
    inertia_values = []
    k_values = range(2, 11)  # Test k from 2 to 10

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[['recency', 'frequency', 'monetary']])
        inertia_values.append(kmeans.inertia_)

    # Plot the Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertia_values, marker='o')
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

# Function to calculate Silhouette Scores and return as DataFrame
def calculate_silhouette_scores(data):
    silhouette_scores = []
    k_values = range(2, 11)  # Test k from 2 to 10

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data[['recency', 'frequency', 'monetary']])
        silhouette_avg = silhouette_score(data[['recency', 'frequency', 'monetary']], cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Create a DataFrame for Silhouette Scores
    silhouette_df = pd.DataFrame({
        "Number of Clusters (k)": k_values,
        "Silhouette Score": silhouette_scores
    })
    return silhouette_df

# Main function to display preprocessing and visualization
def main():
    st.title("Preprocessing and Visualization")

    # Retrieve the dataframe from session state
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        st.error("No dataset found. Please go back and upload a dataset.")
        return

    # Display the dataframe
    st.dataframe(df, use_container_width=True)

    # Calculate RFM data
    monthly_data = calculate_rfm(df)

    # Visualize data BEFORE handling outliers
    st.header("Data Visualization BEFORE Handling Outliers", anchor=False)
    visualize_rfm_data(monthly_data, "Before Handling Outliers")

    # Handle outliers in the RFM metrics
    monthly_data = handle_outliers(monthly_data, 'recency')
    monthly_data = handle_outliers(monthly_data, 'frequency')
    monthly_data = handle_outliers(monthly_data, 'monetary')

    # Visualize data AFTER handling outliers
    st.header("Data Visualization AFTER Handling Outliers", anchor=False)
    visualize_rfm_data(monthly_data, "After Handling Outliers")

    # Normalize the RFM metrics
    monthly_data = normalize_data(monthly_data, ['recency', 'frequency', 'monetary'])
    st.header("Data AFTER normalized", anchor=False)
    st.write(monthly_data)

    # Clustering Section
    st.header("K-Means Clustering", anchor=False)

    # Create tabs for Elbow Method and Silhouette Scores
    tab1, tab2 = st.tabs(["Elbow Method", "Silhouette Scores"])

    with tab1:
        # Plot Elbow Method
        st.subheader("Elbow Method for Optimal k")
        plot_elbow_method(monthly_data)

    with tab2:
        # Calculate and display Silhouette Scores as DataFrame
        st.subheader("Silhouette Scores for Optimal k")
        silhouette_df = calculate_silhouette_scores(monthly_data)
        st.dataframe(silhouette_df)

    # Let the user choose the number of clusters
    st.subheader("Choose the Number of Clusters (k)")
    n_clusters = st.slider("Select k", min_value=2, max_value=10, value=4, step=1)

    # Perform K-Means clustering with the selected k
    clustered_data = perform_kmeans_clustering(monthly_data, n_clusters)

    # Display the clustered data
    st.subheader("Clustered Data")
    st.write(clustered_data)

    # Visualize the clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='recency', y='monetary', hue='cluster', data=clustered_data, palette='viridis', ax=ax)
    ax.set_title("Clusters: Recency vs Monetary")
    ax.set_xlabel("Recency")
    ax.set_ylabel("Monetary")
    st.pyplot(fig)

# Run the main function
if __name__ == "__main__":
    main()
