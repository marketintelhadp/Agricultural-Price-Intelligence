import os
import re
import json

BASE_DATA_DIR = "data/raw/processed"
BASE_MODEL_DIR = "models"
OUTPUT_FILE = "web/config_generated.py"

def extract_keys_from_path(path, base, is_model=False):
    rel_path = os.path.relpath(path, base)
    parts = rel_path.split(os.sep)
    if is_model:
        parts = parts[:-1]  # exclude the filename
    return parts

def build_config(data_root, model_root):
    config = {}

    # Load datasets
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                match = re.search(r'(.+?)_([A-Za-z]+)_dataset\.csv', file)
                if not match:
                    continue

                variety, grade = match.groups()
                keys = extract_keys_from_path(path, data_root)

                try:
                    market, fruit, *rest = keys
                except ValueError:
                    continue

                node = config.setdefault(market, {}).setdefault(fruit, {})
                if rest:
                    location = rest[0]
                    node = node.setdefault(location, {}).setdefault(variety, {})
                else:
                    node = node.setdefault(variety, {})

                node.setdefault(grade, {})["dataset"] = path.replace("\\", "/")

    # Load models
    for root, _, files in os.walk(model_root):
        for file in files:
            if file.endswith(".h5"):
                path = os.path.join(root, file)
                match = re.search(r'lstm_(.+?)_grade_([A-Za-z]+)\.h5', file)
                if not match:
                    continue

                variety, grade = match.groups()
                keys = extract_keys_from_path(path, model_root, is_model=True)

                try:
                    market, *rest = keys
                    if len(rest) >= 1:
                        fruit = "Apple" if "Apple" in rest or "Cherry" in rest else rest[0]
                    else:
                        fruit = "Apple"  # default fallback if path is too short
                except ValueError:
                    continue  # skip if path is irregular

                node = config.setdefault(market, {}).setdefault(fruit, {})
                if len(rest) > 2:
                    location = rest[1]
                    node = node.setdefault(location, {}).setdefault(variety, {})
                else:
                    node = node.setdefault(variety, {})

                node.setdefault(grade, {})["model"] = path.replace("\\", "/")

    return config

# Make sure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Write CONFIG to file
config_data = build_config(BASE_DATA_DIR, BASE_MODEL_DIR)
with open(OUTPUT_FILE, "w") as f:
    f.write("CONFIG = ")
    f.write(json.dumps(config_data, indent=4))
    f.write("\n")

print("✅ config_generated.py successfully created.")
