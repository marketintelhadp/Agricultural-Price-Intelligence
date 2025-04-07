import os
import re
import pandas as pd

def extract_model_metrics(txt_path):
    """Extract MSE, MAE, R2 from result file using regex"""
    with open(txt_path, "r") as file:
        content = file.read()

    rmse = re.search(r"MSE\s*[:=]?\s*([0-9.]+)", content)
    mae = re.search(r"MAE\s*[:=]?\s*([0-9.]+)", content)

    return {
        "MSE": float(rmse.group(1)) if rmse else None,
        "MAE": float(mae.group(1)) if mae else None
    }

def extract_results(root="model_results"):
    records = []

    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file.endswith("_results.txt"):
                file_path = os.path.join(dirpath, file)

                # Parse path parts
                parts = file_path.replace("\\", "/").split("/")
                try:
                    market = parts[1]
                    variety_grade = parts[2]
                    model = parts[3]
                except IndexError:
                    continue  # skip malformed paths

                if "_" in variety_grade:
                    variety, grade = variety_grade.split("_", 1)
                else:
                    variety, grade = variety_grade, "NA"

                metrics = extract_model_metrics(file_path)

                records.append({
                    "Market": market,
                    "Variety": variety,
                    "Grade": grade,
                    "Model": model,
                    **metrics
                })

    df = pd.DataFrame(records)
    df.sort_values(by=["Market", "Variety", "Grade", "Model"], inplace=True)
    df.to_csv("compiled_model_results.csv", index=False)
    print("✅ Model results extracted and saved to 'compiled_model_results.csv'")
    return df

if __name__ == "__main__":
    extract_results()
