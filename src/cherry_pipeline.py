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
    df = pd.read_csv(file_path)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Rename columns to match code references
    df.rename(columns={
        "Min Price": "Min Price (per kg)",
        "Max Price": "Max Price (per kg)",
        "Avg Price": "Avg Price (per kg)"
    }, inplace=True)

    return df

# ------------------------------------------------------------------------------
# 4. Generate Descriptive Statistics
# ------------------------------------------------------------------------------
def generate_descriptive_statistics(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    description_folder = os.path.join(output_folder, "descriptive_statistics")
    os.makedirs(description_folder, exist_ok=True)

    # Since we have one variety, only grades need to be handled
    grades = df['Grade'].unique()
    descriptive_results = {}

    for grade in grades:
        grade_df = df[df['Grade'] == grade]
        stats = grade_df.describe()
        descriptive_results[grade] = stats
        
        stats.to_csv(os.path.join(description_folder, f"{grade}_descriptive_statistics.csv"), index=True)

    return descriptive_results

# ------------------------------------------------------------------------------
# 5. Generate Datasets (one per grade)
# ------------------------------------------------------------------------------
def generate_datasets(df, output_folder):
    """
    Creates daily datasets for each grade (and one variety).
    Reindexes the data so all dates from min to max are covered.
    Forward-fills categorical data, fills numeric NaNs with 0,
    and sets 'Mask' to 1 where 'Min Price (per kg)' > 0, else 0.
    """
    #df['Date'] = pd.to_datetime(df['Date'])
    os.makedirs(output_folder, exist_ok=True)
    output_datasets = {}

    grades = df['Grade'].unique()

    for grade in grades:
        grade_df = df[df['Grade'] == grade].copy()

        start_date = grade_df['Date'].min()
        end_date = grade_df['Date'].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Remove duplicate dates by grouping and taking first row per date
        grade_df = grade_df.groupby('Date').first()

        # Set Date as index and reindex with full date range
        grade_df = grade_df.reindex(full_date_range)
        grade_df.index.name = 'Date'
        grade_df.reset_index(inplace=True)

        # Forward-fill categorical columns
        for col in ['District', 'Market', 'Fruit', 'Variety', 'Grade']:
            if col in grade_df.columns:
                grade_df[col] = grade_df[col].ffill()

        numeric_cols = ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
        for col in numeric_cols:
            if col in grade_df.columns:
                grade_df[col] = pd.to_numeric(grade_df[col], errors='coerce').fillna(0)

        # Create mask column: 1 where 'Min Price (per kg)' > 0, else 0
        if 'Min Price (per kg)' in grade_df.columns:
            grade_df['Mask'] = (grade_df['Min Price (per kg)'] > 0).astype(int)
        else:
            grade_df['Mask'] = 0

        dataset_name = f"Cherry_{grade}"
        output_datasets[dataset_name] = grade_df
        output_file = os.path.join(output_folder, f"{dataset_name}_dataset.csv")
        grade_df.to_csv(output_file, index=False)

    return output_datasets

# ------------------------------------------------------------------------------
# 6. Visualize Data
# ------------------------------------------------------------------------------
def visualize_data(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    exploration_results_folder = os.path.join(output_folder, "data_exploration_results")
    os.makedirs(exploration_results_folder, exist_ok=True)

    # Ensure 'Date' is datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # (A) Distribution of all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        numeric_cols.hist(bins=20, figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(exploration_results_folder, 'numerical_features_distribution.png'))
        plt.close()

    # (B) Distribution of Avg Price by Grade (since Variety is fixed)
    if 'Avg Price (per kg)' in df.columns and 'Grade' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x='Avg Price (per kg)', hue='Grade', kde=True)
        plt.title("Distribution of Avg Price (per kg) by Grade")
        plt.savefig(os.path.join(exploration_results_folder, 'avg_price_distribution_by_grade.png'))
        plt.close()

    # (C) Time-series plots (Grouped by Grade)
    if 'Date' in df.columns and 'Avg Price (per kg)' in df.columns:
        filtered_df = df[df['Mask'] == 1] if 'Mask' in df.columns else df

        for grade, subset in filtered_df.groupby('Grade'):
            if not subset.empty:
                subset = subset.sort_values(by='Date')
                plt.plot(subset['Date'], subset['Avg Price (per kg)'],
                         linestyle='-', marker='o', label=f"Grade {grade}")

        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.xlabel('Date')
        plt.ylabel('Avg Price (per kg)')
        plt.title('Trends by Grade (Filtered)')
        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.savefig(os.path.join(exploration_results_folder, 'trends_by_grade_filtered.png'))
        plt.close()

# ------------------------------------------------------------------------------
# 7. Main Function
# ------------------------------------------------------------------------------
def main():
    file_path = r"data/raw/Narwal Cherr.csv"
    output_folder = r"data/raw/processed/Narwal"
    eda_folder = r"Data_exploration_results/Narwal/cherry"

    # Ensure directories exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(eda_folder, exist_ok=True)

    try:
        df = load_data(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    try:
        output_datasets = generate_datasets(df, output_folder)
        logging.info(f"Datasets saved to: {output_folder}")
    except Exception as e:
        logging.error(f"Error generating datasets: {e}")

    try:
        visualize_data(df, eda_folder)
        logging.info(f"EDA images saved to: {eda_folder}")
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

    try:
        descriptive_statistics = generate_descriptive_statistics(df, eda_folder)
        logging.info(f"Descriptive statistics saved to: {eda_folder}")
    except Exception as e:
        logging.error(f"Error generating descriptive statistics: {e}")

    logging.info("All steps completed successfully!")

# ------------------------------------------------------------------------------
# 8. Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
