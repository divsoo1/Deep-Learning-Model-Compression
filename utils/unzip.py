import zipfile

# Specify the path to the zip file
zip_file_path = "/home/ray/nfs/autolang_storage/projects/divyam/zip/archive.zip"

# Specify the directory where you want to extract the files
extracted_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data"

# Create a ZipFile object
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all files and directories from the zip archive
    zip_ref.extractall(extracted_dir)

print(f"Files extracted to {extracted_dir}")


# xray rsync-up path_to_user.yaml cluster_name /mnt/home/a469864/workspace/archive.zip /home/ray/nfs/autolang_storage/projects/divyam/zip