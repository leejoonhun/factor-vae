from zipfile import ZipFile

import polars as pl

from ._path import DATA_DIR


def extract_zip(locale: str):
    for zip in DATA_DIR.parent.iterdir():
        if zip.suffix == ".zip" and locale in zip.stem:
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


def make_csv(locale: str):
    locale_dir = DATA_DIR / locale
    tmp = pl.DataFrame()
    for file in locale_dir.iterdir():
        try:
            tmp = pl.concat([tmp, pl.read_csv(file)])
        except pl.NoDataError:
            pass
    tmp.write_csv(DATA_DIR / f"{locale}.csv")
    print(f"Saved {locale}.csv")


def main(locale: str):
    if not (DATA_DIR / locale).exists():
        extract_zip(locale)
    make_csv(locale)


if __name__ == "__main__":
    for locale in ["hk", "hu", "jp", "pl", "uk", "us"]:
        main(locale)
