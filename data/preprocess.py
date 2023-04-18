from zipfile import ZipFile

import polars as pl

from .path import DATA_DIR


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
        tmp = pl.DataFrame()
        for file in dir.iterdir():
            try:
                tmp = pl.concat([tmp, pl.read_csv(file)])
            except pl.NoDataError:
                pass
        tmp.write_csv(DATA_DIR / f"{dir.name}.csv")
        print(f"Saved {dir.name}.csv")


def main():
    if not (DATA_DIR / "us").exists():
        extract_zip()
    make_csv()


if __name__ == "__main__":
    main()
