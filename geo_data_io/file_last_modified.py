# walks a directory and returns the file last modified time

# standard libraries
import os
import datetime
from pathlib import Path

# external libraries
import pandas as pd


def get_file_last_modified(root_dir, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_files is None:
        exclude_files = []

    exclude_dirs = set(d.lower() for d in exclude_dirs)
    exclude_files = set(f.lower() for f in exclude_files)

    # List to store rows for Excel
    col_names = ["file_order_id", "Directory", "File Name", "Last Modified"]
    rows = []

    file_order_id = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove excluded directories
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs]

        # Filter excluded files
        filenames = [f for f in filenames if f.lower() not in exclude_files]

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            rows.append((file_order_id, os.path.normpath(dirpath), filename, last_modified))
            file_order_id += 1    
    
    # create a pandas dataframe
    df = pd.DataFrame(data = rows, columns = col_names)

    # extract additional date information
    df['Last Modified'] = pd.to_datetime(df['Last Modified'])
    df['Last Modified Order'] = df['Last Modified'].rank(method = 'dense', ascending=False).astype(int)
    df["lm_ymd"] = df["Last Modified"].dt.strftime("%Y/%m/%d")
    df["lm_ym"] = df["Last Modified"].dt.strftime("%Y/%m")
    df["lm_y"] = df["Last Modified"].dt.strftime("%Y")

    df['file_ext'] = df['File Name'].map(lambda x: Path(x).suffix)

    return df


if __name__ == "__main__":

    directory_to_scan = '.'
    print(os.path.abspath(directory_to_scan))
    exclude_list_dirs = ["node_modules", ".git", "__pycache__", ".ipynb_checkpoints", ".idea", ".vscode", ]
    exclude_list_files = ["Thumbs.db", ".DS_Store", "desktop.ini",
                      ".gitattributes", ".gitignore", "__init__.py"]

    df = get_file_last_modified(
    root_dir=directory_to_scan,    
    exclude_dirs=exclude_list_dirs,
    exclude_files=exclude_list_files
    )

    print(df.head())