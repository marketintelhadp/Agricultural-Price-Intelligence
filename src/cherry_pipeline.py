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
    df = pd.read_excel(file_path)

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
def generate_descriptive_statistics(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    description_folder = os.path.join(output_folder, "descriptive_statistics")
    os.makedirs(description_folder, exist_ok=True)

    descriptive_results = {}

    if 'Variety' not in df.columns or 'Grade' not in df.columns:
        logging.error("Missing 'Variety' or 'Grade' column.")
        return {}

    for (variety, grade), group_df in df.groupby(['Variety', 'Grade']):
        stats = group_df.describe()
        descriptive_results[f"{variety}_{grade}"] = stats

        filename = f"{variety}_{grade}_descriptive_statistics.csv"
        stats.to_csv(os.path.join(description_folder, filename), index=True)
        logging.info(f"Saved descriptive stats: {filename}")

    return descriptive_results


# ------------------------------------------------------------------------------
# 5. Generate Datasets (one per grade)
# ------------------------------------------------------------------------------
def generate_datasets(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_datasets = {}

    if 'Variety' not in df.columns or 'Grade' not in df.columns:
        logging.error("Both 'Variety' and 'Grade' columns are required.")
        return {}

    for (variety, grade), subset_df in df.groupby(['Variety', 'Grade']):
        try:
            logging.info(f"Processing dataset for: {variety} - {grade}")
            grade_df = subset_df.copy()

            start_date = grade_df['Date'].min()
            end_date = grade_df['Date'].max()
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            grade_df = grade_df.groupby('Date').first()
            grade_df = grade_df.reindex(full_date_range)
            grade_df.index.name = 'Date'
            grade_df.reset_index(inplace=True)

            for col in ['District', 'Market', 'Fruit', 'Variety', 'Grade']:
                if col in grade_df.columns:
                    grade_df[col] = grade_df[col].ffill()

            for col in ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']:
                if col in grade_df.columns:
                    grade_df[col] = pd.to_numeric(grade_df[col], errors='coerce').fillna(0)

            grade_df['Mask'] = (grade_df['Min Price (per kg)'] > 0).astype(int)

            if grade_df.empty:
                logging.warning(f"{variety} - {grade} resulted in empty dataset after processing.")
            else:
                dataset_name = f"{variety}_{grade}"
                output_datasets[dataset_name] = grade_df
                output_file = os.path.join(output_folder, f"{dataset_name}_dataset.csv")
                grade_df.to_csv(output_file, index=False)
                logging.info(f"Saved dataset: {output_file}")

        except Exception as e:
            logging.error(f"Error processing {variety} - {grade}: {e}")

    return output_datasets


# ------------------------------------------------------------------------------
# 6. Visualize Data
# ------------------------------------------------------------------------------
def visualize_data(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    exploration_results_folder = os.path.join(output_folder, "data_exploration_results")
    os.makedirs(exploration_results_folder, exist_ok=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # A. Distribution of all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        numeric_cols.hist(bins=20, figsize=(10, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(exploration_results_folder, 'numerical_features_distribution.png'))
        plt.close()

    # B. Distribution of Avg Price by Variety and Grade
    if 'Avg Price (per kg)' in df.columns and 'Grade' in df.columns and 'Variety' in df.columns:
        for (variety, grade), group in df.groupby(['Variety', 'Grade']):
            plt.figure(figsize=(10, 6))
            sns.histplot(data=group, x='Avg Price (per kg)', kde=True)
            plt.title(f"Avg Price Distribution - {variety} ({grade})")
            filename = f"{variety}_{grade}_price_distribution.png"
            plt.savefig(os.path.join(exploration_results_folder, filename))
            plt.close()

    # C. Time-series plots by Variety and Grade
    if 'Date' in df.columns and 'Avg Price (per kg)' in df.columns:
        filtered_df = df[df['Mask'] == 1] if 'Mask' in df.columns else df

        for (variety, grade), subset in filtered_df.groupby(['Variety', 'Grade']):
            if not subset.empty:
                subset = subset.sort_values(by='Date')
                plt.figure(figsize=(12, 6))
                plt.plot(subset['Date'], subset['Avg Price (per kg)'],
                         linestyle='-', marker='o', label=f"{variety} - {grade}")
                plt.xlabel('Date')
                plt.ylabel('Avg Price (per kg)')
                plt.title(f'Trend: {variety} - {grade}')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()

                filename = f"{variety}_{grade}_timeseries.png"
                plt.savefig(os.path.join(exploration_results_folder, filename))
                plt.close()

# ------------------------------------------------------------------------------
# 7. Main Function
# ------------------------------------------------------------------------------
def main():
    file_path = r"data/raw/Shopian Cherry.xlsx"
    output_folder = r"data/raw/processed/Shopian"
    eda_folder = r"Data_exploration_results/Shopian/cherry"

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
