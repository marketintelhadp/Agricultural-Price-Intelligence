import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load the dataset
def load_data(file_path):
    """Load the dataset from the specified Excel file."""
    return pd.read_excel(file_path)

# Function to clean data
def clean_data(df):
    """Perform data cleaning tasks and convert prices to per kg."""
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Check if 'Grade' column is not in the dataframe
    no_grade = 'Grade' not in df.columns
    
    # Check if 'CaseType' column is not in the dataframe
    no_casetype = 'CaseType' not in df.columns
    
    # Convert prices per quintal to prices per kg if CaseType is missing
    if no_casetype and no_grade:
        df[['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']] = df[ 
            ['Min Price', 'Max Price', 'Avg Price'] 
        ] / 100
    else:
        # Handle CaseType and convert prices to per kg using case weights
        case_weights = {'FC': 16, 'HC': 8}

        def price_per_kg(row):
            weight = case_weights.get(row['CaseType'], np.nan)
            if pd.notna(weight):
                return [row['Min Price'] / weight, row['Max Price'] / weight, row['Avg Price'] / weight]
            return [np.nan, np.nan, np.nan]

        df[['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']] = df.apply(
            lambda row: pd.Series(price_per_kg(row)), axis=1
        )
    
    # Drop original price columns and CaseType
    df = df.drop(columns=['CaseType', 'Min Price', 'Max Price', 'Avg Price'], errors='ignore')
    
    # Create a mask column where Min Price (per kg) is greater than 0
    df['Mask'] = (df['Min Price (per kg)'] > 0).astype(int)
    
    return df


# Function to generate descriptive statistics
def generate_descriptive_statistics(df, output_folder):
    """Generate descriptive statistics for each variety."""
    os.makedirs(output_folder, exist_ok=True)
    description_folder = os.path.join(output_folder, "descriptive_statistics")
    os.makedirs(description_folder, exist_ok=True)

    varieties = df['Variety'].unique()
    descriptive_results = {}

    for variety in varieties:
        variety_df = df[df['Variety'] == variety]
        stats = variety_df.describe()
        descriptive_results[variety] = stats
        stats.to_csv(os.path.join(description_folder, f"{variety}_descriptive_statistics.csv"))

    return descriptive_results

# Function to generate datasets for each variety and grade
def generate_datasets(df, output_folder):
    """Generate separate datasets for each variety and grade if present."""
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
                subset.set_index('Date', inplace=True)
                subset = subset.reindex(full_date_range)
                subset.index.name = 'Date'
                subset.reset_index(inplace=True)

                subset[['District', 'Market', 'Fruit', 'Variety', 'Grade']] = subset[
                    ['District', 'Market', 'Fruit', 'Variety', 'Grade']
                ].ffill()

                subset[['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']] = subset[
                    ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
                ].fillna(0)
                subset['Mask'] = (subset['Min Price (per kg)'] > 0).astype(int)

                output_datasets[f"{variety}_{grade}"] = subset
                output_file = os.path.join(output_folder, f"{variety}_{grade}_dataset.csv")
                subset.to_csv(output_file, index=False)
        else:
            variety_df.set_index('Date', inplace=True)
            variety_df = variety_df.reindex(full_date_range)
            variety_df.index.name = 'Date'
            variety_df.reset_index(inplace=True)
            
            variety_df[['District', 'Market', 'Fruit', 'Variety']] = variety_df[
                ['District', 'Market', 'Fruit', 'Variety']
            ].ffill()

            variety_df[['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']] = variety_df[
                ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
            ].fillna(0)
            variety_df['Mask'] = (variety_df['Min Price (per kg)'] > 0).astype(int)

            output_datasets[f"{variety}"] = variety_df
            output_file = os.path.join(output_folder, f"{variety}_dataset.csv")
            variety_df.to_csv(output_file, index=False)

    return output_datasets


def visualize_data(df, output_folder):
    """Generate exploratory visualizations."""
    
    # Create directory for saving plots
    exploration_results_folder = os.path.join(output_folder, "data_exploration_results")
    os.makedirs(exploration_results_folder, exist_ok=True)

    # Ensure 'Date' is a datetime object and sort data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')  # Sort entire dataframe by Date

    # Distribution of numerical features
    df.select_dtypes(include='number').hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(exploration_results_folder, 'numerical_features_distribution.png'))
    plt.close()

    # Distribution by variety
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Avg Price (per kg)', hue='Variety', kde=True)
    plt.title("Distribution of Avg Price (per kg) by Variety")
    plt.savefig(os.path.join(exploration_results_folder, 'avg_price_distribution_by_variety.png'))
    plt.close()
    
    # Check if 'Grade' column exists
    if 'Grade' in df.columns:
        for (variety, grade), subset in df[df['Mask'] == 1].groupby(['Variety', 'Grade']):
            if not subset.empty:
                subset = subset.sort_values(by='Date')  # Ensure each subset is sorted properly
                plt.plot(subset['Date'], subset['Avg Price (per kg)'], linestyle='-', marker='o', label=f"{variety} - {grade}")
    else:
        for variety, subset in df[df['Mask'] == 1].groupby('Variety'):
            if not subset.empty:
                subset = subset.sort_values(by='Date')  # Ensure each subset is sorted properly
                plt.plot(subset['Date'], subset['Avg Price (per kg)'], linestyle='-', marker='o', label=f"{variety}")

    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.xlabel('Date')
    plt.ylabel('Avg Price (per kg)')
    plt.title('Trends by Variety and Grade (Filtered by Mask)' if 'Grade' in df.columns else 'Trends by Variety (Filtered by Mask)')

    # Improve x-axis readability
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exploration_results_folder, 'trends_by_variety_and_grade_filtered.png' if 'Grade' in df.columns else 'trends_by_variety_filtered.png'))
    plt.close()
    
# Main function
def main():
    # File path to the dataset
    file_path = r"D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\raw\NarwalFinal.xlsx"
    output_folder = r"D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\raw\processed\Narwal"
    eda_folder = r"D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\Data_exploration_results\Narwal"
    
    # Load data
    try:
        df = load_data(file_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Clean data
    df = clean_data(df)

    # Generate descriptive statistics
    try:
        descriptive_statistics = generate_descriptive_statistics(df, eda_folder)
    except Exception as e:
        logging.error(f"Error generating descriptive statistics: {e}")

    # Visualize data
    try:
        visualize_data(df, eda_folder)
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

    # Generate datasets
    try:
        output_datasets = generate_datasets(df, output_folder)
    except Exception as e:
        logging.error(f"Error generating datasets: {e}")

    logging.info("Datasets generated, descriptive statistics created, and visualizations completed successfully!")

if __name__ == "__main__":
    main()
