import datetime
import time
from pathlib import Path
import zlib
import os
import proquint
import argparse

def get_checksum(path):
    with open(path, 'rb') as f:
        fb = f.read()
        crc_hash = zlib.crc32(fb)
    return crc_hash

def get_proquint(uint):
    return proquint.uint2quint(uint)

def iso_date():
    return time.strftime("%Y-%m-%d")

def get_modified_iso_date(path):
    modify_time = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(modify_time).isoformat()[:10]

def main(args):
    # rename each video file 
    dir_to_rename = os.path.abspath(args.dir)
    for file in os.listdir(dir_to_rename):
        if file[-3:].lower() in ["mp4", "m4v", "avi", "mkv", "mov"] and file[:1] != '.':
            suffix = file[-3:]
            root = os.path.abspath(dir_to_rename)
            filepath_to_rename = f"{root}/{file}"
            proquint_str = get_proquint(get_checksum(filepath_to_rename))
            new_filepath = f"{root}/{get_modified_iso_date(filepath_to_rename)}-{proquint_str}.{suffix}"
            print(f"Renaming\n\t{filepath_to_rename}\nto \n \t{new_filepath}")
            os.rename(filepath_to_rename, new_filepath)
        else:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename all files in given directory to {ISO format}-{unique hash converted to pronounceable string}.{suffix} .\nUsage: video_renamer.py --dir directory_containing_files_to_rename")
    parser.add_argument('--dir', default='.', type=str, help='Directory to search for video files to rename')
    args = parser.parse_args()
    
    main(args)
