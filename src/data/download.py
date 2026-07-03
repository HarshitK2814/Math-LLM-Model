"""Download the raw datasets as parquet files into data/raw/.

Uses the Hugging Face parquet export endpoints directly, so no HF token or
`datasets` library is required. GSM-Plus-BN lives on Mendeley Data and must be
downloaded manually (see README in data/raw/ after running this script).
"""

from pathlib import Path
from urllib.request import urlretrieve

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

HF_PARQUET = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{path}"

FILES = {
    "bangla_math_train.parquet": HF_PARQUET.format(
        repo="kawchar85/Bangla-Math", path="default/train/0000.parquet"
    ),
    "ganit_sft_train.parquet": HF_PARQUET.format(
        repo="dipta007/Ganit", path="SFT/train/0000.parquet"
    ),
    "ganit_dev.parquet": HF_PARQUET.format(
        repo="dipta007/Ganit", path="dev/dev/0000.parquet"
    ),
}

GSM_PLUS_BN_NOTE = """GSM-Plus-BN is hosted on Mendeley Data and requires a manual download:
https://data.mendeley.com/datasets/74dscnmrhv/2
Place the CSV/XLSX file(s) in this directory as gsm_plus_bn.* and re-run preprocess.
"""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW_DIR / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...")
        urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    (RAW_DIR / "README_GSM_PLUS_BN.txt").write_text(GSM_PLUS_BN_NOTE, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
