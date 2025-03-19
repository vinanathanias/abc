# Function to calculate silhouette scores for different numbers of clusters
def calculate_silhouette_scores(data, max_clusters=10):
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data[['recency', 'frequency', 'monetary']])
        score = silhouette_score(data[['recency', 'frequency', 'monetary']], labels)
        silhouette_scores.append(score)
    return silhouette_scores

# Function to calculate inertia for the elbow method
def calculate_inertia(data, max_clusters=10):
    inertia_values = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[['recency', 'frequency', 'monetary']])
        inertia_values.append(kmeans.inertia_)
    return inertia_values

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

    # Calculate silhouette scores and inertia for different numbers of clusters
    max_clusters = 10
    silhouette_scores = calculate_silhouette_scores(normalized_data, max_clusters)
    inertia_values = calculate_inertia(normalized_data, max_clusters)

    # Create a dataframe for silhouette scores
    silhouette_df = pd.DataFrame({
        'Number of Clusters (k)': range(2, max_clusters + 1),
        'Silhouette Score': silhouette_scores
    })

    # Display silhouette scores in a dataframe
    st.subheader("Silhouette Scores for Different Numbers of Clusters")
    st.write("The table below shows the silhouette scores for different numbers of clusters. "
             "A higher silhouette score indicates better-defined clusters.")
    st.dataframe(silhouette_df, use_container_width=True)

    # Visualize the elbow method
    st.subheader("Elbow Method for Optimal Number of Clusters")
    st.write("The line chart below shows the inertia (sum of squared distances) for different numbers of clusters. "
             "The 'elbow' point indicates the optimal number of clusters.")

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Display the silhouette scores as a line chart
        st.write("**Silhouette Scores Line Chart**")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='b')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Scores for Different k")
        st.pyplot(fig)

    with col2:
        # Display the elbow method plot
        st.write("**Elbow Method Line Chart**")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='-', color='r')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig)

    # Let the user choose the number of clusters
    n_clusters = st.slider("Choose the Number of Clusters (k)", min_value=2, max_value=max_clusters, value=4, step=1)

    # Save the selected number of clusters to session state
    st.session_state.n_clusters = n_clusters

    # Add a submit button to trigger clustering
    if st.button("Submit", on_click=handle_clustering, type="primary"):
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

# Run the main function
if __name__ == "__main__":
    main()
