import os
import pandas as pd

def main():
    base_path = os.path.join("data", "raw", "processed")

    azadpur_files = [
        "Azadpur/Makhmali_Fancy_dataset.csv",
        "Azadpur/Makhmali_Special_dataset.csv",
        "Azadpur/Makhmali_Super_dataset.csv",
        "Azadpur/Misri_Fancy_dataset.csv",
        "Azadpur/Misri_Special_dataset.csv",
        "Azadpur/Misri_Super_dataset.csv"
    ]

    cherry_markets = ["Ganderbal", "Narwal", "Parimpore", "Shopian"]

    cherry_files = []
    for market in cherry_markets:
        cherry_files.extend([
            f"{market}/Cherry_Large_dataset.csv",
            f"{market}/Cherry_Medium_dataset.csv",
            f"{market}/Cherry_Small_dataset.csv"
        ])

    all_files = azadpur_files + cherry_files

    df_list = []
    for file_rel_path in all_files:
        file_path = os.path.join(base_path, file_rel_path)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            temp_df['SourceFile'] = file_rel_path  # Optional: track source file
            df_list.append(temp_df)
        else:
            print(f"Warning: File not found: {file_path}")

    if not df_list:
        print("No datasets loaded. Exiting.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")

    output_path = os.path.join("data", "combined_azadpur_cherry_dataset.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")

if __name__ == "__main__":
    main()
