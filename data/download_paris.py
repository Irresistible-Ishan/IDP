import os
import sys
import time
import requests
import tarfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
url = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris.tar.gz"
file_name = os.path.join(SCRIPT_DIR, "SN3_roads_train_AOI_3_Paris.tar.gz")

def download_with_resume(url, filename):
    headers = {}
    downloaded = 0
    if os.path.exists(filename):
        downloaded = os.path.getsize(filename)
        headers['Range'] = f"bytes={downloaded}-"
    
    try:
        print(f"Resuming download from {downloaded / (1024*1024):.2f} MB...")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 416: 
            print("File is fully downloaded.")
            return True
            
        if response.status_code not in (200, 206):
            print(f"Error: HTTP {response.status_code}")
            return False

        if response.status_code == 200:
            downloaded = 0
            mode = 'wb'
        else:
            mode = 'ab'

        total_length = int(response.headers.get('content-length', 0)) + downloaded
        
        with open(filename, mode) as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_length > 0:
                        percent = (downloaded / total_length) * 100
                        sys.stdout.write(f"\rDownloading... {percent:.2f}% ({downloaded / (1024*1024):.1f} MB / {total_length / (1024*1024):.1f} MB)")
                        sys.stdout.flush()
        print("\nDownload chunk finished.")
        
        if total_length > 0 and downloaded >= total_length:
            return True
        return False
    except Exception as e:
        print(f"\nConnection dropped: {e}")
        return False

print("Starting robust downloader for Full Paris Training Dataset (5.6 GB)...")
while True:
    success = download_with_resume(url, file_name)
    if success:
        print("\nVerifying and extracting archive... (This will take a few minutes for 5.6 GB)")
        try:
            with tarfile.open(file_name, "r:gz") as tar:
                tar.extractall(path=SCRIPT_DIR)
            print("Extraction complete!")
            break
        except Exception as e:
            print(f"Extraction failed (might be incomplete): {e}. Retrying download in 5s...")
            time.sleep(5)
    else:
        print("Retrying in 5 seconds...")
        time.sleep(5)
