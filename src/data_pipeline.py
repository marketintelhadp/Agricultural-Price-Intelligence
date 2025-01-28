import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_data(file_path):
    """Load the dataset from the specified Excel file."""
    return pd.read_excel(file_path)

# Function to clean data
def clean_data(df):
    """Perform data cleaning tasks and convert prices to per kg."""
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle CaseType and convert prices to per kg
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
    """Generate separate datasets for each variety and grade."""
    varieties = df['Variety'].unique()
    grades = df['Grade'].unique()
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    os.makedirs(output_folder, exist_ok=True)
    output_datasets = {}

    for variety in varieties:
        for grade in grades:
            subset = df[(df['Variety'] == variety) & (df['Grade'] == grade)].copy()

            # Align data with the full date range
            subset.set_index('Date', inplace=True)
            subset = subset.reindex(full_date_range)
            subset.index.name = 'Date'
            subset.reset_index(inplace=True)

            # Fill forward relevant fields
            subset[['District', 'Market', 'Fruit', 'Variety', 'Grade']] = subset[
                ['District', 'Market', 'Fruit', 'Variety', 'Grade']
            ].ffill()

            # Replace NaN prices with 0 and set mask
            subset[['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']] = subset[
                ['Min Price (per kg)', 'Max Price (per kg)', 'Avg Price (per kg)']
            ].fillna(0)
            subset['Mask'] = (subset['Min Price (per kg)'] > 0).astype(int)

            # Save dataset
            output_datasets[f"{variety}_{grade}"] = subset
            output_file = os.path.join(output_folder, f"{variety}_{grade}_dataset.csv")
            subset.to_csv(output_file, index=False)

    return output_datasets

# Function for data visualization
def visualize_data(df, output_folder):
    """Generate exploratory visualizations."""
    # Create directory for saving plots
    exploration_results_folder = os.path.join(output_folder, "data_exploration_results")
    os.makedirs(exploration_results_folder, exist_ok=True)

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

    # Trends over time (using Mask = 1)
    plt.figure(figsize=(14, 8))
    for (variety, grade), subset in df[df['Mask'] == 1].groupby(['Variety', 'Grade']):
        if not subset.empty:
            plt.plot(subset['Date'], subset['Avg Price (per kg)'], label=f"{variety} - {grade}")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel('Date')
    plt.ylabel('Avg Price (per kg)')
    plt.title('Trends by Variety and Grade (Filtered by Mask)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exploration_results_folder, 'trends_by_variety_and_grade_filtered.png'))
    plt.close()

# Main function
def main():
    # File path to the dataset
    file_path = r"D:\ML Repositories\Price_forecasting_project\data\raw\Pulwama Prichoo Final.xlsx"
    output_folder = r"D:\ML Repositories\Price_forecasting_project\data\raw\processed\Pulwama\Prichoo"

    # Load data
    df = load_data(file_path)

    # Clean data
    df = clean_data(df)

    # Generate descriptive statistics
    descriptive_statistics = generate_descriptive_statistics(df, output_folder)

    # Visualize data
    visualize_data(df, output_folder)

    # Generate datasets
    output_datasets = generate_datasets(df, output_folder)

    print("Datasets generated, descriptive statistics created, and visualizations completed successfully!")

if __name__ == "__main__":
    main()
