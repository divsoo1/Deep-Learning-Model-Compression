import zipfile

zip_file_path = "/home/ray/nfs/autolang_storage/projects/divyam/zip/archive.zip"

extracted_dir = "/home/ray/nfs/autolang_storage/projects/divyam/data"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

print(f"Files extracted to {extracted_dir}")
