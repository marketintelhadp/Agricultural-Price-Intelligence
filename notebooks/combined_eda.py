import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load combined dataset
    file_path = "data/combined_azadpur_cherry_dataset.csv"
    df = pd.read_csv(file_path)

    # Remove SourceFile column if exists
    if 'SourceFile' in df.columns:
        df = df.drop(columns=['SourceFile'])

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Basic info
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    print("Descriptive statistics:\n", df.describe(include='all'))

    # Distribution of Avg Price (per kg) by Variety and Market
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Avg Price (per kg)', hue='Variety', multiple='stack', kde=True)
    plt.title("Distribution of Average Price (per kg) by Variety")
    plt.show()

    # Boxplot of Avg Price (per kg) by Market, Variety and Grade
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Market', y='Avg Price (per kg)', hue='Grade', data=df)
    plt.title("Boxplot of Average Price (per kg) by Market and Grade")
    plt.xticks(rotation=45)
    plt.show()

    # Countplot for Market, Variety and Grade
    plt.figure(figsize=(16, 8))
    sns.countplot(x='Market', hue='Grade', data=df)
    plt.title("Countplot of Market and Grade")
    plt.xticks(rotation=45)
    plt.show()

    # Missing values heatmap by Market, Variety and Grade
    missing_values = df.groupby(['Market', 'Variety', 'Grade']).apply(lambda x: x.isnull().sum())
    plt.figure(figsize=(14, 7))
    sns.heatmap(missing_values, annot=True, cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap by Market, Variety and Grade")
    plt.show()

    # Time series trends of Avg Price (per kg) by Market, Variety and Grade
    df_sorted = df.sort_values('Date')
    df_sorted['Sequence'] = df_sorted.groupby(['Market', 'Variety', 'Grade']).cumcount() + 1

    plt.figure(figsize=(16, 10))
    grouped = df_sorted.groupby(['Market', 'Variety', 'Grade'])
    for (market, variety, grade), subset in grouped:
        if not subset.empty:
            plt.plot(subset['Sequence'], subset['Avg Price (per kg)'], label=f"{market} - {variety} - {grade}")
    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Observation Number')
    plt.ylabel('Average Price (per kg)')
    plt.title('Price Trends by Market, Variety and Grade')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
