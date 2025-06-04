# scripts/download_kkg_data.py

import os
import time
import glob
import pandas as pd  # Added import for pandas
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

# === SETTINGS ===
username = "shaheen_skuast"
password = "Kkg@1234"
download_dir = os.path.abspath(r"D:\Git Projects\Price_forecasting_project\Agricultural-Price-Intelligence\data\real_time")

# === USER INPUT ===
valid_crops = ["Apple", "Cherry"]
print("Available crops:", ", ".join(valid_crops))
crop_choice = input("Enter crop name (Apple/Cherry): ").strip().title()

if crop_choice not in valid_crops:
    print(f"❌ Invalid crop: {crop_choice}. Exiting...")
    exit(1)



# === CONFIGURE SELENIUM ===
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True,  # <= helps Chrome allow custom download dirs
    "safebrowsing.disable_download_protection": True  # optional, extra safety
})
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

service = Service(executable_path="scripts/chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 20)

try:
    # Step 1: Open login page
    driver.get("https://kkg.jk.gov.in/kkg/login/")
    print("🔐 Opened login page...")

    # Step 2: Click Login button (modal trigger)
    login_button = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "button.custom_btn.green_btn"))
    )
    login_button.click()
    print("➡️ Clicked Login button")

    # Step 3: Enter username
    username_input = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//input[contains(@placeholder, 'Username') or contains(@placeholder, 'mobile')]")
    ))
    username_input.send_keys(username)

    wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button[contains(text(),'SUBMIT')]"))).click()
    print("✅ Submitted username")

    # Step 4: Enter password
    password_input = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//input[@type='password']")
    ))
    password_input.send_keys(password)

    wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button[contains(text(),'SUBMIT')]"))).click()
    print("🔐 Submitted password — please solve CAPTCHA manually")

    # Step 5: Wait for dashboard to fully load
    print("⏳ Waiting for dashboard to load...")
    while True:
        if "dashboard" in driver.current_url:
            try:
                # Wait until dropdown is present and enabled
                dropdown = wait.until(EC.presence_of_element_located(
                    (By.XPATH, "//select[contains(@class, 'chakra-select')]")
                ))
                wait.until(lambda d: dropdown.is_enabled())

                # Wait until table is populated (at least one row or not showing '0 records')
                wait.until(EC.presence_of_element_located(
                    (By.XPATH, "//table//tbody/tr")
                ))
                print("✅ Dashboard loaded and ready.")
                break
            except:
                print("⏳ Still waiting for dropdown/table to load...")
        time.sleep(2)

    # Step 6: Select "Apple" from dropdown
    print(f"🌾 Selecting '{crop_choice}' from crop dropdown...")
    select = Select(dropdown)
    select.select_by_visible_text(crop_choice)

    time.sleep(2)
    
    # Step 7: Click export icon
    print("📁 Clicking export icon...")
    
    # Locate the export icon based on its SVG path and click its parent
    export_icon = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button//*[name()='svg' and contains(@viewBox, '0 0 576 512')]/ancestor::button")
    ))
    export_icon.click()
    
    # Step 8: Click 'CSV' from dropdown
    print("⬇️ Clicking 'CSV' option...")
    csv_button = wait.until(EC.element_to_be_clickable((
        By.XPATH,
        "//button[.//text()[normalize-space()='CSV']]"
    )))
    csv_button.click()
    print("✅ CSV export clicked. Waiting for download...")

    time.sleep(10)  # Allow time for download
    
    # Check download folder for CSV file
    csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
    if csv_files:
        print(f"📄 CSV Downloaded: {csv_files[0]}")
    else:
        print("⚠️ No CSV file found in download folder!")
        
    def wait_for_download_and_rename(folder, crop_name, timeout=30):
        print("⏳ Waiting for CSV download to complete...")
        seconds = 0
        while seconds < timeout:
            files = os.listdir(folder)

            # Wait if download is still in progress
            if any(f.endswith(".crdownload") for f in files):
                time.sleep(1)
                seconds += 1
                continue

            # Check for completed CSV file (with known default name)
            csv_files = [f for f in files if f.endswith(".csv") and "Daily Crop" in f]
            if csv_files:
                # Get full path of the latest matching file
                csv_file = sorted(
                    [os.path.join(folder, f) for f in csv_files],
                    key=os.path.getmtime,
                    reverse=True
                )[0]

                try:
                    # Read the CSV
                    df = pd.read_csv(csv_file)

                    # Convert epoch to readable date format
                    df['Date/Time'] = pd.to_datetime(df['Date/Time'], unit='s').dt.strftime('%Y-%m-%d')

                    # Generate unique filename
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = os.path.join(folder, f"{crop_name.lower()}_kkg_data_{timestamp_str}.csv")

                    # Save processed CSV
                    df.to_csv(new_name, index=False)
                    print(f"📦 CSV saved as: {new_name}")

                    # Delete the original downloaded file
                    os.remove(csv_file)
                    print(f"🗑️ Deleted temp file: {csv_file}")

                    return new_name

                except Exception as e:
                    print(f"❌ Error processing CSV file: {e}")
                    return None

            time.sleep(1)
            seconds += 1

        print("⚠️ Timeout: CSV file not downloaded or detected.")
        return None


    # === Trigger and process the downloaded file ===
    result_file = wait_for_download_and_rename(download_dir, crop_choice)
    if result_file:
        print("✅ Download and processing completed successfully.")
    else:
        print("❌ Failed to download or process the CSV.")


except Exception as e:
    print(f"❌ ERROR: {e}")
    driver.save_screenshot("error.png")
    with open("page_debug.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print("📄 Saved screenshot and HTML for debugging.")

finally:
    driver.quit()
    print("✅ Done. Browser closed.")
