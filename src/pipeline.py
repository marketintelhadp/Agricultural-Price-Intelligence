import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ------------------------------------------------------------------------------
# 1. Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# 2. Load the Dataset
# ------------------------------------------------------------------------------
def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Rename price columns for consistency (no conversion needed)
    df.rename(columns={
        "Min Price": "Min Price (per kg)",
        "Max Price": "Max Price (per kg)",
        "Avg Price": "Avg Price (per kg)"
    }, inplace=True)
    return df

# ------------------------------------------------------------------------------
# 3. Generate Descriptive Statistics (Variety + Grade)
# ------------------------------------------------------------------------------
def generate_descriptive_statistics(df, output_folder):
    """Generate stats for each variety and grade."""
    os.makedirs(output_folder, exist_ok=True)
    description_folder = os.path.join(output_folder, "descriptive_statistics")
    os.makedirs(description_folder, exist_ok=True)

    descriptive_results = {}
    varieties = df['Variety'].unique()

    for variety in varieties:
        variety_df = df[df['Variety'] == variety]
        grades = variety_df['Grade'].unique()

        for grade in grades:
            grade_df = variety_df[variety_df['Grade'] == grade]
            stats = grade_df.describe()
            key = f"{variety}_{grade}"
            descriptive_results[key] = stats
            stats.to_csv(os.path.join(description_folder, f"{key}_descriptive_statistics.csv"), index=True)

    return descriptive_results

# ------------------------------------------------------------------------------
# 4. Generate Datasets (One per Variety + Grade)
# ------------------------------------------------------------------------------
def generate_datasets(df, output_folder):
    """Split data by variety and grade, reindex dates, and forward-fill."""
    os.makedirs(output_folder, exist_ok=True)
    output_datasets = {}
    varieties = df['Variety'].unique()

    for variety in varieties:
        variety_df = df[df['Variety'] == variety]
        grades = variety_df['Grade'].unique()

        for grade in grades:
            grade_df = variety_df[variety_df['Grade'] == grade].copy()
            start_date = grade_df['Date'].min()
            end_date = grade_df['Date'].max()
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Remove duplicate dates
            grade_df = grade_df.groupby('Date').first().reset_index()

            # Reindex to full date range
            grade_df.set_index('Date', inplace=True)
            grade_df = grade_df.reindex(full_date_range)
            grade_df.index.name = 'Date'
            grade_df.reset_index(inplace=True)

            # Forward-fill categorical columns
            categorical_cols = ['District', 'Market', 'Fruit', 'Variety', 'Grade']
            grade_df[categorical_cols] = grade_df[categorical_cols].ffill()

            # Fill numeric NaNs with 0 and add mask
            numeric_cols = ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
            grade_df[numeric_cols] = grade_df[numeric_cols].fillna(0)
            grade_df['Mask'] = (grade_df['Min Price (per kg)'] > 0).astype(int)

            # Save dataset
            dataset_name = f"{variety}_{grade}"
            output_datasets[dataset_name] = grade_df
            output_file = os.path.join(output_folder, f"{dataset_name}_dataset.csv")
            grade_df.to_csv(output_file, index=False)

    return output_datasets

# ------------------------------------------------------------------------------
# 5. Visualize Data (Variety + Grade)
# ------------------------------------------------------------------------------
def visualize_data(df, output_folder):
    """Generate plots for each variety and grade."""
    os.makedirs(output_folder, exist_ok=True)
    exploration_results_folder = os.path.join(output_folder, "data_exploration_results")
    os.makedirs(exploration_results_folder, exist_ok=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # (A) Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        numeric_cols.hist(bins=20, figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(exploration_results_folder, 'numerical_features_distribution.png'))
        plt.close()

    # (B) Avg Price distribution by Variety + Grade
    if 'Avg Price (per kg)' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x='Avg Price (per kg)', hue='Variety', kde=True, element='step')
        plt.title("Avg Price Distribution by Variety")
        plt.savefig(os.path.join(exploration_results_folder, 'avg_price_distribution_by_variety.png'))
        plt.close()

    # (C) Time-series trends (filtered by Mask)
    if 'Avg Price (per kg)' in df.columns:
        filtered_df = df[df['Mask'] == 1] if 'Mask' in df.columns else df

        for (variety, grade), subset in filtered_df.groupby(['Variety', 'Grade']):
            subset = subset.sort_values(by='Date')
            plt.plot(subset['Date'], subset['Avg Price (per kg)'], 
                    linestyle='-', marker='o', label=f"{variety} ({grade})")

        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.xlabel('Date')
        plt.ylabel('Avg Price (per kg)')
        plt.title('Price Trends by Variety and Grade')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(exploration_results_folder, 'trends_by_variety_grade.png'))
        plt.close()

# ------------------------------------------------------------------------------
# 6. Main Function
# ------------------------------------------------------------------------------
def main():
    file_path = "data/raw/Azadpur Cherry.csv"
    output_folder = "data/raw/processed/Azadpur"
    eda_folder = "Data_exploration_results/Azadpur"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(eda_folder, exist_ok=True)

    try:
        df = load_data(file_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    try:
        output_datasets = generate_datasets(df, output_folder)
        logging.info(f"Saved datasets to: {output_folder}")
    except Exception as e:
        logging.error(f"Error generating datasets: {e}")

    try:
        visualize_data(df, eda_folder)
        logging.info(f"Saved visualizations to: {eda_folder}")
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

    try:
        generate_descriptive_statistics(df, eda_folder)
        logging.info(f"Saved descriptive stats to: {eda_folder}")
    except Exception as e:
        logging.error(f"Error generating stats: {e}")

    logging.info("Pipeline completed!")

if __name__ == "__main__":
    main()