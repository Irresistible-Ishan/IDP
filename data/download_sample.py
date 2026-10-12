import urllib.request
import tarfile
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
url = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs/SN3_roads_sample.tar.gz"
file_name = os.path.join(SCRIPT_DIR, "SN3_roads_sample.tar.gz")

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}%")
    sys.stdout.flush()

print(f"Downloading {file_name} from SpaceNet AWS bucket...")
urllib.request.urlretrieve(url, file_name, reporthook=download_progress)
print("\nDownload complete. Extracting...")

with tarfile.open(file_name, "r:gz") as tar:
    tar.extractall(path=SCRIPT_DIR)
print("Extraction complete!")
