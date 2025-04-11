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
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Rename columns to match code references
    # Make sure these names match your CSV exactly
    df.rename(columns={
        "Min Price": "Min Price (per kg)",
        "Max Price": "Max Price (per kg)",
        "Avg Price": "Avg Price (per kg)"
    }, inplace=True)

    return df

# ------------------------------------------------------------------------------
# 3. Minimal "Clean" Step (Remove duplicates) -- We'll handle Mask in generate_datasets
# ------------------------------------------------------------------------------
def clean_data(df):
    """
    Removes duplicates. We won't create a Mask here, we'll do it in generate_datasets.
    """
    df = df.drop_duplicates()
    return df

# ------------------------------------------------------------------------------
# 4. Generate Descriptive Statistics
# ------------------------------------------------------------------------------
def generate_descriptive_statistics(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    description_folder = os.path.join(output_folder, "descriptive_statistics")
    os.makedirs(description_folder, exist_ok=True)

    varieties = df['Variety'].unique()
    descriptive_results = {}

    for variety in varieties:
        variety_df = df[df['Variety'] == variety]
        stats = variety_df.describe()
        descriptive_results[variety] = stats
        
        stats.to_csv(os.path.join(description_folder, f"{variety}_descriptive_statistics.csv"), index=True)

    return descriptive_results

# ------------------------------------------------------------------------------
# 5. Generate Datasets (one per Variety, optionally split by Grade)
# ------------------------------------------------------------------------------
def generate_datasets(df, output_folder):
    """
    Creates daily datasets for each variety (and each grade if 'Grade' exists).
    Reindexes the data so all dates from min to max are covered.
    Forward-fills categorical data, then sets 'Mask' to 1 only if all three numeric columns 
    have nonzero values, and finally fills numeric NaNs with 0.
    """
    varieties = df['Variety'].unique()
    os.makedirs(output_folder, exist_ok=True)
    output_datasets = {}

    for variety in varieties:
        variety_df = df[df['Variety'] == variety].copy()

        start_date = variety_df['Date'].min()
        end_date = variety_df['Date'].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        if 'Grade' in df.columns:
            grades = variety_df['Grade'].unique()
            for grade in grades:
                subset = variety_df[variety_df['Grade'] == grade].copy()

                # Reindex on full date range
                subset.set_index('Date', inplace=True)
                subset = subset.reindex(full_date_range)
                subset.index.name = 'Date'
                subset.reset_index(inplace=True)

                # Forward-fill text columns
                subset[['District', 'Market', 'Fruit', 'Variety', 'Grade']] = subset[
                    ['District', 'Market', 'Fruit', 'Variety', 'Grade']
                ].ffill()

                numeric_cols = ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']

                # Convert numeric columns explicitly to float (if not already)
                for col in numeric_cols:
                    if col in subset.columns:
                        subset[col] = pd.to_numeric(subset[col], errors='coerce')

                # Compute Mask based on nonzero values BEFORE filling NaNs:
                if all(col in subset.columns for col in numeric_cols):
                    mask_condition = (
                        (subset['Min Price (per kg)'] != 0) &
                        (subset['Max Price (per kg)'] != 0) &
                        (subset['Avg Price (per kg)'] != 0)
                    )
                    subset['Mask'] = mask_condition.astype(int)
                else:
                    subset['Mask'] = 0

                # Now fill numeric NaNs with 0 (this won't affect the computed mask)
                for col in numeric_cols:
                    if col in subset.columns:
                        subset[col] = subset[col].fillna(0)

                dataset_name = f"{variety}_{grade}"
                output_datasets[dataset_name] = subset
                output_file = os.path.join(output_folder, f"{dataset_name}_dataset.csv")
                subset.to_csv(output_file, index=False)
        else:
            # Handle case when there's no Grade column
            variety_df.set_index('Date', inplace=True)
            variety_df = variety_df.reindex(full_date_range)
            variety_df.index.name = 'Date'
            variety_df.reset_index(inplace=True)

            variety_df[['District', 'Market', 'Fruit', 'Variety']] = variety_df[
                ['District', 'Market', 'Fruit', 'Variety']
            ].ffill()

            numeric_cols = ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
            for col in numeric_cols:
                if col in variety_df.columns:
                    variety_df[col] = pd.to_numeric(variety_df[col], errors='coerce')

            if all(col in variety_df.columns for col in numeric_cols):
                mask_condition = (
                    (variety_df['Min Price (per kg)'] != 0) &
                    (variety_df['Max Price (per kg)'] != 0) &
                    (variety_df['Avg Price (per kg)'] != 0)
                )
                variety_df['Mask'] = mask_condition.astype(int)
            else:
                variety_df['Mask'] = 0

            for col in numeric_cols:
                if col in variety_df.columns:
                    variety_df[col] = variety_df[col].fillna(0)

            output_datasets[variety] = variety_df
            output_file = os.path.join(output_folder, f"{variety}_dataset.csv")
            variety_df.to_csv(output_file, index=False)

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

    # (B) Distribution of Avg Price by Variety
    if 'Avg Price (per kg)' in df.columns and 'Variety' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x='Avg Price (per kg)', hue='Variety', kde=True)
        plt.title("Distribution of Avg Price (per kg) by Variety")
        plt.savefig(os.path.join(exploration_results_folder, 'avg_price_distribution_by_variety.png'))
        plt.close()

    # (C) Time-series plots
    if 'Date' in df.columns and 'Avg Price (per kg)' in df.columns:
        # If there's a Grade column, group by variety & grade
        if 'Grade' in df.columns:
            for (variety, grade), subset in df[df['Mask'] == 1].groupby(['Variety', 'Grade']):
                if not subset.empty:
                    subset = subset.sort_values(by='Date')
                    plt.plot(subset['Date'], subset['Avg Price (per kg)'],
                             linestyle='-', marker='o', label=f"{variety} - {grade}")
        else:
            # Otherwise group by variety only
            for variety, subset in df[df['Mask'] == 1].groupby('Variety'):
                if not subset.empty:
                    subset = subset.sort_values(by='Date')
                    plt.plot(subset['Date'], subset['Avg Price (per kg)'],
                             linestyle='-', marker='o', label=variety)

        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.xlabel('Date')
        plt.ylabel('Avg Price (per kg)')
        plt.title('Trends by Variety and Grade (Filtered by Mask)' 
                  if 'Grade' in df.columns else 
                  'Trends by Variety (Filtered by Mask)')
        
        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = ('trends_by_variety_and_grade_filtered.png'
                    if 'Grade' in df.columns 
                    else 'trends_by_variety_filtered.png')
        plt.savefig(os.path.join(exploration_results_folder, filename))
        plt.close()

# ------------------------------------------------------------------------------
# 7. Main Function
# ------------------------------------------------------------------------------
def main():
    file_path = r"D:\ML Repositories\Price_forecasting_project\data\raw\Shopian_Cherry.csv"
    output_folder = r"D:\ML Repositories\Price_forecasting_project\data\raw\processed\Shopian"
    eda_folder = r"D:\ML Repositories\Price_forecasting_project\Data_exploration_results\Shopian\data_exploration_results\cherry"

    try:
        df = load_data(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    df = clean_data(df)

    try:
        descriptive_statistics = generate_descriptive_statistics(df, eda_folder)
    except Exception as e:
        logging.error(f"Error generating descriptive statistics: {e}")

    try:
        visualize_data(df, eda_folder)
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

    try:
        output_datasets = generate_datasets(df, output_folder)
    except Exception as e:
        logging.error(f"Error generating datasets: {e}")

    logging.info("Datasets generated, descriptive statistics created, and visualizations completed successfully!")

# ------------------------------------------------------------------------------
# 8. Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
