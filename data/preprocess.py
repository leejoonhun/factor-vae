from pathlib import Path
from zipfile import ZipFile

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def extract_zip():
    for zip in DATA_DIR.parent.iterdir():
        if zip.suffix == ".zip":
            locale = zip.stem.split("_")[1]
            with ZipFile(zip, "r") as zip_file:
                zip_file.extractall(DATA_DIR)
            CURRENT_DIR = DATA_DIR / "data" / "daily" / locale
            DEST_DIR = DATA_DIR / locale
            DEST_DIR.mkdir(exist_ok=True)
            for dir in CURRENT_DIR.iterdir():
                for file in dir.iterdir():
                    if file.is_dir():
                        for f in file.iterdir():
                            if f.name.split(".")[-2] == locale:
                                _ = f.replace(f"{DEST_DIR}/{f.name}")
                    else:
                        if file.name.split(".")[-2] == locale:
                            _ = file.replace(f"{DEST_DIR}/{file.name}")
            (DATA_DIR / "data").rmdir()
    print("Extracted zip files")


def make_csv():
    for dir in DATA_DIR.iterdir():
        tmp = pd.DataFrame()
        for file in dir.iterdir():
            try:
                tmp = pd.concat([tmp, pd.read_csv(file)], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass
        tmp.to_csv(DATA_DIR / f"{dir.name}.csv", index=False)
        print(f"Saved {dir.name}.csv")


def main():
    if not (DATA_DIR / "us").exists():
        extract_zip()
    make_csv()


if __name__ == "__main__":
    main()
