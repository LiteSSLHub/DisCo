import os
import json
import logging
import csv

import torch
from pyarrow.json import read_json

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def get_logger(name):
    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    logger = logging.getLogger(name)

    return logger


def read_jsonl(fp):
    raw = read_json(fp)
    return WarpJsonObject(raw)


def write_jsonl(fp, data):
    with open(fp, "w") as f:
        f.writelines([json.dumps(it) + "\n" for it in data])


def read_tsv(input_file, quotechar=None, delimiter="\t"):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


class WarpJsonObject:
    """convert pyarrow.Table to python dict"""

    def __init__(self, table):
        self._table = table
        self._feats = table.column_names

        self._start_idx = 0

    def __len__(self):
        return len(self._table)

    def __getitem__(self, index: int):
        return {k: self._table[k][index].as_py() for k in self._feats}

    def __iter__(self):
        self._start_idx = -1
        return self

    def __next__(self):
        self._start_idx += 1

        if self._start_idx == len(self):
            raise StopIteration

        return self[self._start_idx]


def save_checkpoints(filename, ckpt):
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, "wb") as f:
        torch.save(ckpt, f)


def load_checkpoints(filename, device):
    obj = torch.load(filename, map_location=torch.device(device))
    return obj


def parameter_amount(model):
    amount = 0
    for n, p in model.named_parameters():
        amount += p.nelement()
    return amount


def clean_web_text(st):
    """clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        # print("before:\n", st)
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1:]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
        # print("after\n", st)
        # print("")
    st = st.replace("\\n", " ")
    st = st.replace("\\", " ")
    # while "  " in st:
    #   st = st.replace("  ", " ")
    return st


def cnn_dataset_to_jsonl(fp, dataset):
    dict_list = []
    for item in dataset:
        dict_list.append({"src": item[2], "tgt": item[3], "labels": item[1]})
    write_jsonl(fp, dict_list)
