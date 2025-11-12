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


