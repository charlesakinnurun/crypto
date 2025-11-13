# %% [markdown]
# Import the libraries

# %%
import cctx
import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # For 3D plots if needed
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

# %% [markdown]
# Data Fetching

# %%
def fetch_crypto_data(tickers,timeframe="1d",limit=365):
    """ Fetch OHLCV data for a list of tickers from Binance
        Args:
        tickers (list): List of ticker symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
        timeframe (str): The timeframe for candles (e.g., '1d', '4h', '1h').
        limit (int): The number of candles to fetch.

    Returns:
        dict: A dictionary where keys are tickers and values are pandas DataFrames
              of their OHLCV data.
    """

    print(f"Starting data fetch for {len(tickers)} tickers......")
    # Initialize the binance exchange interface
    exchange = cctx.binance()

    # Standard columns names for OHLCV data
    columns = ["timestamp","open","high","low","close","volume"]

    all_data = {}

    for ticker in tickers:
        try:
            # Fetch the data
            # cctx returns a list of lists
            # [[timestamp,open,high,low,close,volume]]
            ohlcv = exchange.fetch_ohlcv(ticker,timeframe,limit=limit)

            # Convert the list of lists to a pandas DataFame
            df = pd.DataFrame(ohlcv,columns=columns)

            # Convert timestamp (milliseconds) to a readable datatime format
            df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms")

            # Store the DataFrame in our dictionary
            all_data[ticker] = df
            print(f"Successfully fetched {ticker}")

            # Be polite to the API: wait a bit before the next request
            # This helps avoid rate limits
            time.sleep(exchange.ratelimit / 1000) # (ratelimit is in ms)
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    print("Data fetch complete")
    return all_data

# %% [markdown]
# Feature Engineering

# %%
def engineer_features(data_dict):
    """
    Engineers features for each crypto to be used for clustering.
    We are clustering the *cryptocurrencies themselves*, so we need
    to aggregate the time-series data into a single row of features
    for each crypto.
    
    Args:
        data_dict (dict): The dictionary of DataFrames from fetch_crypto_data.

    Returns:
        pd.DataFrame: A DataFrame where each row is a crypto and
                      each column is an engineered feature.
    """
    print("Engineering features")
    features = []

    for ticker,df in data_dict.items():
        if df.empty:
            continue
        # 1. Daily Returns: (Close - Open) / Open
        # We calculate the average daily return
        daily_return = (df["close"] - df["open"]) / df["open"]
        avg_daily_return = daily_return.std()

        # 2. Average Daily Volume (in terms of the quote currency, eg USDT)
        # We approximate this by (close * volume)
        avg_daily_volume = (df["close"] * df["volume"]).mean()

        # 3. Volatility: Standard deviation of daily returns
        # This measures how risky or unstable the asset is
        volatility = daily_return.std()

        # 4. Average Daily Range:(High - Low) / Close
        # This measures the average intraday price swing
        daily_range = (df["high"] - df["low"]) / df["close"]
        avg_daily_range = daily_range.mean()

        # Append the features for this ticker to our list
        features.append({
            "ticker":ticker,
            "avg_daily_return":avg_daily_return,
            "volatility":volatility,
            "avg_daily_volume":avg_daily_volume,
            "avg_daily_range":avg_daily_range
        })


    # Convert the lsit of dictionaries to a DataFrame
    feature_df = pd.DataFrame(features)

    # Set the ticker as the index
    feature_df = feature_df.set_index("ticker")

    # Drop any rows with missing data (e.g, if a crypto had no volume)
    feature_df = feature_df.dropna()

    print("Feature engineering complete")
    return feature_df

# %% [markdown]
# Data Scaling

# %%
def preprocess_data(df):
    """
    Scales the feature DataFrame.
    
    Args:
        df (pd.DataFrame): The feature-engineered DataFrame.

    Returns:
        tuple: (scaled_data, scaler_object)
               The scaled data (numpy array) and the scaler
               object (to transform new data later).
    """
    print("Scaling data..........")
    # Clustering algorithms (K-Means,DBSCAN) are distance-based
    # This means features large scales (like "avg_daily_volume")
    # will dominate features with small scales (like "avg_daily_return")
    # We MUST scale the data. StandardScaler transform data to have
    # a mean of 0 and standard deviation of 1

    scaler = StandardScaler()

    # .fit_transform() calculates the mean/std and applies the scaling
    scaled_data = scaler.fit_transform(df)

    return scaled_data,scaler

# %% [markdown]
# Pre-Training Visualization

# %%
def visualize_before(scaled_data,feature_df):
    # Visualizes the dataset before clustering using PCA
    # PCA (principal component analysis) reduces our 4 features
    # down to 2, allowing us to plot them on a 2D scatter plot

    print("Visualizing data before clustering (using PCA).......")

    # Initialize PCA to reduce to 2 components (for 2D plotting)
    pca = PCA(n_components=2)

    # Fot and transform the scaled data
    data_pca = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10,7))
    plt.scatter(data_pca[:,0],data_pca[:,1])

    # Add labels for each point
    for i,ticker in enumerate(feature_df.index):
        plt.annotate(ticker,(data_pca[i,0],data_pca[i,1]),
                     textcoords="offset points",xytext=(0,5),ha="center",fontsize=8)
        
    plt.title("Cryptocurrency Data (Before Clustering) - PCA View")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

    # Return the PCA object and data for later
    return pca,data_pca

# %% [markdown]
# K-Means Clustering (Centriod-Based)

# %%
def tune_and_run_kmeans(scaled_data):
    # Finds the optimal "k" and runs K-Means Clustering

    print("----- K-Means Clustering -----")

    # Hyperparameter Tuning: Finding the best "k"

    # Method 1: The Elbow Method
    # We plot the "Within-Cluster" Sum of Squares (WCSS) for 
    # different values of k. The "elbow" is the point where
    # WCSS starts to decrease less dramatically
    wcss = []
    k_range = range(1,11) # Test k from 1 to 10


    for k in k_range:
        kmeans = KMeans(n_clusters=k,init="k-means++",random_state=42,n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_) # inertia_ is the WCSS


    plt.figure(figsize=(12,6))
    plt.Subplot(1,2,1)
    plt.plot(k_range,wcss,"bo-")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")


    # Method 2: Silhouette Analysis
    # The Silhouette Score measures how similar a point is to its
    # own cluster compared to other clusters
    # Score ranges from -1 (bad) to +1 (good)
    silhouette_scores = []
    k_range_sil = range(2,11) # Silhouette score needs at least 2 clusters


    for k in k_range_sil:
        kmeans = KMeans(n_clusters=k,init="k-means++",random_state=42,n_init=10)
        kmeans.fit(scaled_data)
        score = silhouette_score(scaled_data,kmeans.labels_)
        silhouette_scores.append(score)

    plt.Subplot(1,2,2)
    plt.plot(k_range_sil,silhouette_scores,"rs-")
    plt.title("K-Means Silhoutte Scores")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhoutte Score")
    plt.show()



    # Analysis
    # Based on the plot, we choose the best "k"
    # For the elbow, it's often 3 or 4
    # For Silhouette, we pick the highest score
    # Let's assume the plots suggest k=4 is a good balance
    optimal_k = 3 # This should be chosen based on your plots
    print(f"Based on analysis (Elbow & Silhouette), choosing k = {optimal_k}")


    # ----- Final Model Training -----
    kmeans_model = KMeans(n_clusters=optimal_k,init="k-means++",random_state=42,n_init=10)
    kmeans_model.fit(scaled_data)

    # Get the cluster labels for each data point
    kmeans_labels = kmeans_model.labels_

    # Evaluate the final model
    sil_score = silhouette_score(scaled_data,kmeans_labels)
    db_score = davies_bouldin_score(scaled_data,kmeans_labels)

    print(f"K-Means (k={optimal_k}) Silhouette Score: {sil_score:.4f}")
    print(f"K-Means (k={optimal_k}) Davies-Bouldin Score: {db_score:.4f}")


    return kmeans_model,kmeans_labels,(sil_score,db_score)

# %% [markdown]
# Agglomerative Clustering (Hierarchial)

# %%
from scipy.cluster.hierarchy import dendrogram,linkage

def tune_and_run_agglo(scaled_data):
    # Visualizes the dendogram and runs Agglomerative Clustering

    print("-----  Agglomerative Clustering -----")

    # Hyperparameter Tuning: The Dendrogram
    # The dendrogram is a tree diagram that shows the hierarchical
    # relationships. We "tune" it by visually inspecting it
    # to find the best place to "cut" the tree

    # "ward" linkage minimizes the variance of the clusters being merged
    linked = linkage(scaled_data,method="ward")

    plt.figure(figsize=(12,7))
    dendrogram(linked,
               orientation="top",
               labels=feature_df.index,
               distance_sort="descending",
               show_leaf_counts=True)
    plt.title("Agglomerative Clustering Dendrogram")
    plt.xlabel("Cryptocurrency")
    plt.ylabel("Distance (Ward)")
    plt.axhline(y=3.5,color="r",linestyle="'--") # Add a line to show a potential cut
    plt.show()


    # ----- Analysis -----
    # Look for the longest vertical lines that are not cut by
    # a horizontal line. Cutting at y=3.5 (red line) would
    # result in 4 clsuters Let's use n_clusters=4
    optimal_n = 4 # This should be chosen based on the denndrogram
    print(f"Based on dendrogram, choosing n_clusters = {optimal_n}")

    # ----- Final Model Training -----
    agglo_model = AgglomerativeClustering(n_clusters=optimal_n,linkage="ward")

    # .fit_predict() trains and returns the labels
    agglo_labels = agglo_model.fit_predict(scaled_data)

    # Evaluate the final model
    sil_score = silhouette_score(scaled_data,agglo_labels)
    db_score = davies_bouldin_score(scaled_data,agglo_labels)

    print(f"Agglomerative (n={optimal_n}) Silhouette Score: {sil_score:.4f}")
    print(f"Agglomerative (n={optimal_n}) Davies-Bouldin Score: {db_score:.4f}")


    return agglo_model,agglo_labels,(sil_score,db_score)

# %% [markdown]
# DBSCAN Clustering (Density-Based)

# %%
def tune_and_run_dbscan(scaled_data):
    # Finds optimal "eps" and runs DBSCAN
    print("----- DBSCAN Clustering -----")

    # Hyperparameter Tuning: Finding "eps" and min_samples

    # 1. "min_samples" : A good rule of thumb is (2 * num_features)
    # We have 4 features, so min_samples = 8 is a good start
    min_samples = 2 * scaled_data.shape[1]

    # 2. "eps": To find "eps", we plot the distance of each point to
    # its k-th nearest neighbor (where k = min_samples)
    # We look for the "knee" or "elbow" in the plot

    # Find the distance to the min_samples-th neighbor
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neighbors.fit(scaled_data)
    distances,indices = nbrs.kneighbors(scaled_data)

    # Get the distance to  the k-th neigbour (index min_samples-1)
    # and sort them
    k_distances = np.sort(distances[:,min_samples-1],axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(k_distances)
    plt.title(f"K-Distance Plot  (k={min_samples})")
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel(f"{min_samples}-th Neighbor Distance (eps)")
    plt.axhline(y=1.5,color="r",linestyle="--") # Add a line for the "knee"
    plt.show()

    # ----- Analysis -----
    # The plot shows a sharp bend (knee) around y=1.5
    # This suggests a good value for "eps"
    optimal_eps = 1.5 # This should be chosen based on the plot
    print(f"Based on k-distance plot, choosing eps = {optimal_eps}")
    print(f"Using min_samples = {min_samples}")

    # ----- Final Model Training -----
    dbscan_model = DBSCAN(eps=optimal_eps,min_samples=min_samples)
    dbscan_labels = dbscan_model.fit_predict(scaled_data)

    # DBSCAN labels:
    # -1: Noise (Outlier)
    # 0,1,2, ........ Cluster labels
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")

    # ----- Evaluation -----
    # IMPORTANT: We must exclude noise points (-1)  when calculating
    # metrics like Silhouette,as they don't belong to a cluster

    # Check if more than one cluster was found (excluding noise)
    if n_clusters > 1:
        # Create a mask to select only non-noise points
        non_noise_mask = (dbscan_labels != -1)

        # Selects the data and labels for non-noise points
        data_filtered = scaled_data[non_noise_mask]
        labels_filtered = dbscan_labels[non_noise_mask]

        sil_score = silhouette_score(data_filtered,labels_filtered)
        db_score = davies_bouldin_score(data_filtered,labels_filtered)

        print(f"DBSCAN (non-noise) Silhoette Score: {sil_score:.4f}")
        print(f"DBSCAN (non-noise) Davies-Bouldin Score: {db_score:.4f}")

    else:
        print("DBSCAN did not find more than one cluster. Metrics cannot be calculated")
        sil_score,db_score = np.nan, np.nan


    return dbscan_model,dbscan_labels, (sil_score,db_score)

# %% [markdown]
# Post-Training Visualization

# %%
def visualize_comparison(data_pca,labels_dict,metrics_dict,feature_df):
    # Create a side-by-side plot comparing the clustering results
    print("\nVisualizing Clustering Comparison........")

    fig,axes = plt.subplots(1,3,figsize=(24,8),sharex=True,sharey=True)
    plt.suptitle("Clustering Model Comparison (on 2D  PCA Data)",fontsize=20,y=1.02)

    plot_map = {
        "K-Means":labels_dict["Kmeans"],
        "Agglomerative":labels_dict["agglo"],
        "DBSCAN":labels_dict["dbscan"]
    }


    for i, (title,labels) in enumerate(plot_map.items()):
        ax = axes[1]

        # Use a consistent color palette
        # Create a palette that includes -1 (for noise) if present
        unique_labels = sorted(list(set(labels)))
        palette = sns.color_palette("deep",len(unique_labels))

        # Make noise points (-1) black
        color_map = {label: color for label,color in zip(unique_labels,palette)}
        if -1 in color_map:
            color_map[-1] = (0,0,0) # Black for noise

        # Map labels to colors
        colors = [color_map[label] for label in labels]

        ax.scatter(data_pca[:,0], data_pca[:,1], c=colors)

        # Add metrics to the title
        sil = metrics_dict[title][0]
        db = metrics_dict[title][1]
        ax.set_title(f"{title}",
                     f"Silhouette:{sil:.4f} (Higher is better)",
                     f"Davies-Bouldin: {db:.4f} (Lower is better)",
                     fontsize=14)
        ax.set_ylabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")

    # Add labels (can get crowded, so we do it once)
    for i, ticker in enumerate(feature_df.index):
        axes[1].annotate(ticker,(data_pca[i,0],data_pca[i,1]),
                         textcoords="offset points",xytext=(0.5),
                         ha="center",fontsize=8,color="gray")
    plt.tight_layout()
    plt.show()


    # ----- Print Final Report -----
    print("----- Final Model Comparison -----")
    print("Silhouette Score (Higher is better, range -1 to 1):")
    for model,(sil,db) in metrics_dict.items():
        print(f"    - {model}: {db:.4f}")


    print("Davies Bouldin Score (Lower is better, min 0):")
    for model, (sil,db) in metrics_dict.items():
        print(f"   - {model}: {db:.4f}")


    # ----- Which is best ? ------
    # This is subjective!
    # K-Means and Agglomerative force all points into a cluster
    # DBSCAN is great for outlier detectio (the noise points)

    # We can programmatically find the "best" based on metrics
    best_sil = max(metrics_dict.items(),key=lambda item: item[1][0])
    best_db = min(metrics_dict.items(),key=lambda item: item[1][1])

    print("----- Conclusion -----")
    print(f"Best by Silhouette Score: {best_sil[0]} ({best_db[1][0]:.4f})")
    print(f"Best by Davies-Bouldin Score: {best_db[0]} ({best_db[1][1]:.4f})")


    if best_sil[0] == best_db[0]:
        print(f"Overall, {best_sil[0]} appears to be strongest model quantitatively")
    else:
        print(f"The metrics disagree. {best_sil[0]} is best by separation, but {best_db[0]}")


    print("Qualitative Analysis")
    print("K-Means/Agglomerative: Good for segmenting all assets into groups (e.g, high-risk,stable)")
    print("DBSCAN: Good for finding normal assets and outliers (the noise points)")

# %% [markdown]
# New Prediction Function

# %%
def predict_new_crypto(new_ohlcv_data, scaler, kmeans_model):
    """
    Predicts the cluster for a new, unseen cryptocurrency.
    We will use the K-Means model as it's the most common
    for new predictions.
    
    Args:
        new_ohlcv_data (list): A list of [timestamp, o, h, l, c, v]
                               lists, just like ccxt returns.
        scaler (StandardScaler): The *original* scaler object
                                 we saved from preprocessing.
        kmeans_model (KMeans): The *original* trained K-Means model.

    Returns:
        int: The predicted cluster label.
    """

    print("----- New Prediction -----")
    # 1. Convert to DataFrame
    df = pd.DataFrame(new_ohlcv_data,columns=["timestamp","open","high","low","close","volume"])

    # 2. Wrap in a dictioanry for the feature engineer
    data_dict = {"NEW CRYPTO":df}

    # 3. Engineer Features
    # This will reeturn a 1-row DataFrame
    feature_df = engineer_features(data_dict)

    # 4. Scale the Data
    # We use .transform(), NOT .fit_transform()!
    # We must apply the same scaling as the original training data
    scaled_feature = scaler.transform(feature_df)

    print(f"New Crypto Features (Scaled): {scaled_feature}")

    # 5. Predict
    cluster = kmeans_model.predict(scaled_feature)


    return cluster[0]

# %% [markdown]
# Main Execution

# %%
if __name__ == "__main__":
    
    # --- Define Tickers ---
    # We want a good mix of assets
    crypto_tickers = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'LINK/USDT',
        'MATIC/USDT', 'XRP/USDT', 'LTC/USDT', 'ADA/USDT', 'AVAX/USDT',
        'BNB/USDT', 'TRX/USDT', 'DOT/USDT', 'UNI/USDT', 'SHIB/USDT'
    ]
    
    # --- Step 1 & 2 ---
    raw_data_dict = fetch_crypto_data(crypto_tickers)
    feature_df = engineer_features(raw_data_dict)
    
    # Check the engineered features
    print("\n--- Engineered Features ---")
    print(feature_df.head())
    
    # --- Step 3 ---
    scaled_data, data_scaler = preprocess_data(feature_df)
    
    # --- Step 4 ---
    pca_obj, data_pca = visualize_before(scaled_data, feature_df)
    
    # --- Step 5, 6, 7 ---
    # We will store the models, labels, and metrics for comparison
    models = {}
    all_labels = {}
    all_metrics = {}
    
    # K-Means
    models['kmeans'], all_labels['kmeans'], all_metrics['K-Means'] = \
        tune_and_run_kmeans(scaled_data)
        
    # Agglomerative
    models['agglo'], all_labels['agglo'], all_metrics['Agglomerative'] = \
        tune_and_run_agglo(scaled_data)

    # DBSCAN
    models['dbscan'], all_labels['dbscan'], all_metrics['DBSCAN'] = \
        tune_and_run_dbscan(scaled_data)
        
    # --- Step 8 ---
    visualize_comparison(data_pca, all_labels, all_metrics, feature_df)
    
    # --- Step 9 ---
    # Create some mock data for a new, very stable coin
    # [timestamp, open, high, low, close, volume]
    mock_new_data = [
        [1678886400000, 1.00, 1.01, 1.00, 1.01, 100000],
        [1678972800000, 1.01, 1.01, 1.00, 1.00, 120000],
        [1679059200000, 1.00, 1.02, 1.00, 1.01, 110000],
        [1679145600000, 1.01, 1.01, 1.01, 1.01, 90000]
    ] # ... and so on for 365 days.
    
    # For this example, let's just use data from a real coin
    # to prove the function works. We'll use BTC's data.
    print("\n--- Testing Prediction Function (using BTC data as a 'new' coin) ---")
    btc_raw_data = raw_data_dict['BTC/USDT'][['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
    
    predicted_cluster = predict_new_crypto(
        new_ohlcv_data=btc_raw_data,
        scaler=data_scaler,
        kmeans_model=models['kmeans']
    )
    
    print(f"\nThe 'new' crypto (BTC data) was placed in K-Means cluster: {predicted_cluster}")
    
    # Let's see which cluster BTC was *actually* in
    btc_original_cluster = all_labels['kmeans'][feature_df.index.get_loc('BTC/USDT')]
    print(f"The original BTC was in cluster: {btc_original_cluster}")
    
    if predicted_cluster == btc_original_cluster:
        print("Success! The prediction function correctly clustered the new data.")
    else:
        print("Note: The cluster is different. This can happen due to minor data differences.")


