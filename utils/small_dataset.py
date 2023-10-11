import os
import shutil
import random
from .paths import DATA_DIR, DATA_DIR_SMALL

source_dir = DATA_DIR
destination_dir = DATA_DIR_SMALL

split_ratio = 0.5

def create_smaller_dataset(src_dir, dest_dir, split_ratio):
    for folder_name in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir, folder_name)):
            dest_sub_dir = os.path.join(dest_dir, folder_name)
            os.makedirs(dest_sub_dir, exist_ok=True)

            file_list = os.listdir(os.path.join(src_dir, folder_name))
            num_files = len(file_list)
            num_samples = int(num_files * split_ratio)

            selected_files = random.sample(file_list, num_samples)

            for file_name in selected_files:
                src_file = os.path.join(src_dir, folder_name, file_name)
                dest_file = os.path.join(dest_sub_dir, file_name)
                shutil.copy(src_file, dest_file)


create_smaller_dataset(os.path.join(source_dir, 'train'), os.path.join(destination_dir, 'small_train'), split_ratio)
create_smaller_dataset(os.path.join(source_dir, 'test'), os.path.join(destination_dir, 'small_test'), split_ratio)
create_smaller_dataset(os.path.join(source_dir, 'valid'), os.path.join(destination_dir, 'small_valid'), split_ratio)