from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def preprocess():
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
    preprocess()


if __name__ == "__main__":
    main()
