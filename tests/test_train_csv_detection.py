import csv
import os
from pathlib import Path
from gpu_rl_trading.training.train import load_eurusd_csv


def write_csv(path, delimiter):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>"])
        writer.writerow(["2021.01.01", "00:00:00", "1.10000", "1.10050", "1.09950", "1.10010", "100"])


def test_load_eurusd_csv_comma(tmp_path):
    p = tmp_path / "test_comma.csv"
    write_csv(p, ',')
    arr = load_eurusd_csv(str(p))
    assert arr.shape[1] == 5


def test_load_eurusd_csv_tab(tmp_path):
    p = tmp_path / "test_tab.tsv"
    write_csv(p, '\t')
    arr = load_eurusd_csv(str(p))
    assert arr.shape[1] == 5
