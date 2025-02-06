import os
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to log system status
def log_system_status():
    uptime = time.time() - start_time
    logging.info(f"System Uptime: {uptime / 60:.2f} minutes")
    logging.info(f"Current Time: {datetime.now()}")
    
# Function to monitor directory for new data
def monitor_directory(directory_path):
    logging.info(f"Monitoring directory: {directory_path}")
    previous_files = set(os.listdir(directory_path))
    
    while True:
        time.sleep(60)  # Check every minute
        current_files = set(os.listdir(directory_path))
        
        new_files = current_files - previous_files
        if new_files:
            logging.info(f"New files detected: {', '.join(new_files)}")
        
        previous_files = current_files

# Main function
def main():
    # Set directory to monitor (replace with your directory)
    directory_to_monitor = "data/raw/processed/Pulwama/Prichoo"
    
    # Start monitoring
    log_system_status()
    monitor_directory(directory_to_monitor)

if __name__ == "__main__":
    start_time = time.time()  # Track system start time
    main()
