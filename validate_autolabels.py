from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt

def main(args):
    total_files_checked = 0
    problem_files = 0
    for subdir in Path(args.dir).glob('*'):
        autolabel_dir = Path(subdir)/'labels'
        stills_dir = Path(subdir)/'clean_images'
        for label_txt in autolabel_dir.glob('*'):
            print(f"Checking: {label_txt}")
            lines = 0
            with open(label_txt, 'r') as file:
                total_files_checked += 1
                for line in file:
                    lines += 1
            if lines == 0:
                problem_files += 1
                if not args.dry_run:
                    os.remove(label_txt)
                print(f"EMPTY FILE: {label_txt} to be removed")

                corresponding_still_path = label_txt.parent.parent/'clean_images'/label_txt.name
                if not args.dry_run:
                    os.remove(corresponding_still_path)
                print(f"and corresponding image to be removed: {corresponding_still_path}")

        for still_file in stills_dir.glob('*'):
            print(f"Checking: {still_file}")
            try:
                plt.imread(still_file)
                total_files_checked += 1
            except:
                problem_files += 1
                if not args.dry_run:
                    os.remove(still_file)
                print(f"UNREADABLE IMAGE: {still_file} to be removed")

                corresponding_label_path = still_file.parent.parent/'labels'/still_file.name
                if not args.dry_run:
                    os.remove(corresponding_label_path)
                print(f"and coresponding label text to be removed: {corresponding_label_path}")
    print(f"Done. Total files checked: {total_files_checked}")
    if args.dry_run:
        print("Dry run. Nothing was actually removed.")
    else:
        print(f"{problem_files} empty or corrupt files and their associated files have been removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check that text files are not empty and that image files are readable.\nUsage: python3 validate_autolabels.py --dir data/output_frames")
    parser.add_argument('--dir', default='output', type=str, help="Path to where autolabel.py saves outputs. This directory should contain directories for each video, which will contain 'autolabel' and 'stills' directories")
    parser.add_argument('--dry-run', default=False, action='store_true', help="Don't actually remove anything")
    args = parser.parse_args()
    
    main(args)
