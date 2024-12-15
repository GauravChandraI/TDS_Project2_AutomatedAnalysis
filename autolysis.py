# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "seaborn",
#   "matplotlib",
#   "openai",
#   "uv",
#   "httpx",
#   "pandas",
#   "requests",
#   "numpy",
#   "scikit-learn",
#   "scipy",
#   "python-dotenv",
#   "tiktoken",
# ]
# ///

import os
import sys
import tiktoken
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import ChatCompletion
import openai
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
load_dotenv()

# Ensure the environment variable for AI Proxy token is set, by setting token in your .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Initialize OpenAI client, here in our case, we'll use aiproxy api key and api url, one can get their own proxy token using iitm email id.
PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

# Helper function to save visualizations
def save_plot(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")

# Helper function to Load CSV file
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path) 
        print("CSV file loaded successfully.")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1') # I am using two different encoding methods in case the the try statement fails because of UnicodeDecodeError
            print("CSV file loaded successfully.")
            return df
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='Windows-1252')
            print("CSV file loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def analyze_dataset(df,output_dir, IsLowOnTokens = False):
    analysis_results = {}

    # Basic information
    analysis_results['shape'] = df.shape
    analysis_results['columns'] = [
        {'name': col, 'type': str(df[col].dtype), 'example_values': df[col].dropna().unique()[:5].tolist()}
        for col in df.columns
    ]

    # Summary statistics
    analysis_results['summary_statistics'] = df.describe(include='all').to_dict()
    
    # Missing values
    analysis_results['missing_values'] = df.isnull().sum().to_dict()

    # Correlation matrix
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        threshold = 0.7

        if IsLowOnTokens : # If we are low on token, so we need to reduce the analysis results json, so I have here simplified the correlation_matrix 
            correlation_matrix = correlation_matrix[correlation_matrix.abs() > threshold]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Correlation Matrix")
        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
        save_plot(fig,output_dir, "correlation_matrix.png")
        analysis_results['correlation_matrix'] = correlation_matrix.to_dict()
    
    # Histograms
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(df[numeric_columns[0]], kde=True, ax=ax)
        ax.set_title(f"Distribution of {numeric_columns[0]}")
        ax.set_xlabel(numeric_columns[0])
        ax.set_ylabel("Frequency")
        save_plot(fig,output_dir, f"{numeric_columns[0]}_distribution.png")
    
    # Outlier detection
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_cols.dropna(axis=1, how='all')))
    outliers = (z_scores > 3).sum(axis=0)
    analysis_results['outliers'] = outliers.to_dict()

    # Clustering (if applicable)
    if numeric_cols.shape[1] >= 2:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(numeric_cols.fillna(numeric_cols.mean()))
        df['Cluster'] = clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=numeric_cols.columns[0], y=numeric_cols.columns[1], hue='Cluster', palette='viridis', ax=ax)
        ax.set_title("KMeans Clustering")
        ax.set_xlabel(numeric_cols.columns[0])
        ax.set_ylabel(numeric_cols.columns[1])
        save_plot(fig,output_dir, "clusters.png")
        analysis_results['kmeans_inertia'] = kmeans.inertia_

    # Regression analysis
    if numeric_cols.shape[1] > 1:
        target_col = numeric_cols.columns[-1]
        feature_cols = numeric_cols.columns[:-1]
        model = LinearRegression()
        model.fit(df[feature_cols].fillna(0), df[target_col].fillna(0))

        predictions = model.predict(df[feature_cols].fillna(0))
        coefficients = {col: coef for col, coef in zip(feature_cols, model.coef_)}
        analysis_results['regression_analysis'] = {
            'target': target_col,
            'coefficients': coefficients,
            'intercept': model.intercept_
        }

        # Scatter plot of actual vs predicted values
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df[target_col], y=predictions, ax=ax)
        ax.set_title(f"Regression Analysis: Actual vs Predicted ({target_col})")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        save_plot(fig, output_dir, "regression_actual_vs_predicted.png")
 

    # Time series analysis
    time_cols = []
    for col in df.columns:
        try:
            # Suppress UserWarning while checking if the column is a datetime column
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                converted_col = pd.to_datetime(df[col], errors='coerce')
            
            # Check if all values are valid datetime values
            if converted_col.notnull().all():
                time_cols.append(col)
        except Exception:
            pass
    if time_cols:
        time_col = time_cols[0]  # Use the first time-like column
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col)
        analysis_results['time_series'] = {
            'time_column': time_col,
            'start_date': df[time_col].min().isoformat(),
            'end_date': df[time_col].max().isoformat(),
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        df.set_index(time_col).plot(ax=ax)
        ax.set_title("Time Series Plot")
        ax.set_xlabel("Time")
        ax.set_ylabel("Values")
        save_plot(fig, output_dir, "time_series_plot.png")

    # Geographic analysis
    geo_cols = [col for col in df.columns if 'lat' in col.lower() or 'lon' in col.lower()]
    if len(geo_cols) >= 2:
        lat_col, lon_col = geo_cols[:2]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=lon_col, y=lat_col, ax=ax)
        ax.set_title("Geographic Scatterplot")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        save_plot(fig, output_dir, "geographic_scatterplot.png")

    return analysis_results

def get_chat_completion(model,messages):
    # Define the payload i.e our model and messages object
    payload = {
        "model": model,
        "messages": messages
    }

    try:
        # Make the POST request to the proxy API
        response = requests.post(PROXY_URL, headers=HEADERS, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during API call: {e}")
        return None


# Narrate story
def narrate_story(df,file_name, output_dir,analysis_results):
    try:
        model="gpt-4o-mini"
        try:
            analysis_results_json = json.dumps(analysis_results)
            tokenizer = tiktoken.encoding_for_model(model)
            total_tokens = len(tokenizer.encode(analysis_results_json))

            # Test if tokens count are way too much or not
            if total_tokens >= 30000:
                analysis_results = analyze_dataset(df,output_dir,IsLowOnTokens=True)
                try:
                    analysis_results_json = json.dumps(analysis_results)

                    total_tokens = len(tokenizer.encode(analysis_results_json))
                    if total_tokens >= 30000:
                        del analysis_results["columns"]['example_values'] #Example values are way too much tokens oriented
                        try:
                            analysis_results_json = json.dumps(analysis_results)
                        except: 
                            analysis_results_json = analysis_results
                except: 
                    analysis_results_json = analysis_results
        except: 
            analysis_results_json = analysis_results
        
        messages=[
            {"role": "system", "content": 
                                f"""You are a data scientist narrating the story of a dataset for {file_name}, your goal is to properly narrate a story of the dataset taking into account of the provided dataset analysis json results. 
                                    You will be provided the dataset analyis as a json object which has various details about the dataset along with examples of the dataset inorder to provide proper dataset analyis.
                                    The json fields provided are dataframe shape which provides information about the number of rows and columns in the dataset, 
                                    Columns name of the dataset, Summary statistics of the dataset and the missing values dictionary.
                                    Along with these basic analysis, you will also be provided some detailed analysis fields as per the dataset like correlation_matrix as dictionary, 
                                    outliers dictionary in dataset, kmeans_inertia value of kmeans clustering, regression analysis and timeseries informations if the dataset is applicable to that.
                                    Note that these detailed features are dataset dependent, so these will not always be present in the json, but if they are present make sure to include this information while narrating the story.
                                    Additionally, include references to the visualizations that accompany the analysis."""},
            {"role": "user", "content": f"Here's the analysis of the dataset: {analysis_results_json}"}
        ]

        story = get_chat_completion(model,messages) 
        
    except Exception as e:
        story = f"Error generating narrative: {e}"

    chart_files = [
        f for f in os.listdir(output_dir)
        if f.endswith(".png")
    ]
    visualization_section = "\n\n## Generated Visualizations\n\n"
    for chart in chart_files:
        visualization_section += f"![{chart}]({chart})\n\n"

    # Combine the narrative and visualizations
    full_readme_content = f"# Dataset Analysis for {file_name}\n\n{story}{visualization_section}"


    # Save story to README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(full_readme_content)

    print("Analysis story saved as README.md")


def analyze_and_generate_output(file_path):
    # Define output directory based on file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = load_csv(file_path)
        if df is not None:  
            # Analyze dataset
            analysis_results = analyze_dataset(df,output_dir)
            print("Analysis completed.")

            # Narrate the story
            narrate_story(df, base_name, output_dir,analysis_results)

            return output_dir
        else:
            print(f"None Dataframe")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

# Main workflow
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py dataset.csv")
        sys.exit(1)

    file_paths = sys.argv[1:]
    output_dirs = []

    # Process each dataset file
    for file_path in file_paths:
        if os.path.exists(file_path):
            output_dir = analyze_and_generate_output(file_path)
            output_dirs.append(output_dir)
        else:
            print(f"File {file_path} not found!")

    print(f"Analysis completed. Results saved in directories: {', '.join(output_dirs)}")


if __name__ == "__main__":
    main()
