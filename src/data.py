import os
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer
from collections import namedtuple
import numpy as np
import json
import logging
from tqdm import tqdm

from utils import (
    read_jsonl,
    cnn_dataset_to_jsonl,
    read_tsv,
    get_logger,
    clean_web_text
)

log = get_logger(__name__)
InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


# Example classes used for classification tasks such as agnews, yahoo, dbpedia, imdb, yelp-2, yelp-5, and others.
class UDAData(object):
    def __init__(self, text=None, label=None):
        self.text = text
        self.label = label


# Convert examples from classification tasks such as agnews, yahoo, dbpedia, imdb, yelp-2, yelp-5 into features.
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        '''
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        '''
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label = label_map[example.label] if example.label != -1 else -1

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        segment_ids = torch.tensor(segment_ids)
        label = torch.tensor(label)

        features.append((input_ids, input_mask, segment_ids, label, example.text))
    return features


# Used for the General Distill phase, convert the used examples into features.
def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    # truncate
    if len(tokens) > max_seq_length:
        log.info('len(tokens): {}'.format(len(tokens)))
        log.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        log.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    # The preprocessed data should be already truncated
    assert len(tokens) == len(segment_ids) <= max_seq_length

    # The reason for directly using convert_tokens_to_ids without the need to add [CLS] and [SEP] as in the task_distill
    # part is because the tokens have already been added during the execution of pregenerate_training_data.py.
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    # padding
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


# Parameters required for processing the CNN-DailyMail dataset.
def get_dataset_info(name):
    if name == "cnndm":
        dataset_info = {
            "text_max_len": 512,
            "max_text_ntokens_per_sent": 200,
            "min_text_ntokens_per_sent": 5,
        }

    return dataset_info


# DataModule used for the extractive summarization task, responsible for generating
# the dataloaders required during the training and evaluation phases.
class ConsistSumDataModule:
    def __init__(
            self,
            dataset_name: str,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            tokenizer_name_or_path: str,
            num_workers: int = 0,
    ):
        self.train_filename = os.path.join("./data", dataset_name, "train.jsonl")
        self.val_filename = os.path.join("./data", dataset_name, "val.jsonl")
        self.test_filename = os.path.join("./data", dataset_name, "test.jsonl")

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)

        dataset_info = get_dataset_info(dataset_name)
        self.text_max_len = dataset_info["text_max_len"]
        self.max_text_ntokens_per_sent = dataset_info["max_text_ntokens_per_sent"]
        self.min_text_ntokens_per_sent = dataset_info["min_text_ntokens_per_sent"]

        self.collator = partial(collator,
                                cls_id=self.tokenizer.cls_token_id,
                                sep_id=self.tokenizer.sep_token_id,
                                pad_id=self.tokenizer.pad_token_id,
                                text_max_len=self.text_max_len)

    def prepare(self, supervised_dataset_size=None):
        text_args = {
            "text_max_len": self.text_max_len,
            "max_text_ntokens_per_sent": self.max_text_ntokens_per_sent,
            "min_text_ntokens_per_sent": self.min_text_ntokens_per_sent,
        }
        self.train_dataset = ConsistSumDataset(self.train_filename, self.tokenizer, **text_args)
        self.val_dataset = ConsistSumDataset(self.val_filename, self.tokenizer, **text_args)
        self.test_dataset = ConsistSumDataset(self.test_filename, self.tokenizer, **text_args)

        assert supervised_dataset_size <= len(
            self.train_dataset), f"The total size of the dataset is {len(self.train_dataset)}!"
        log.info(
            f"Split train dataset: {supervised_dataset_size} for supervised; data {len(self.train_dataset)} for unsupervised data.")
        supervised_size, unsupervised_size = supervised_dataset_size, len(self.train_dataset) - supervised_dataset_size
        
        # Use a portion of the training set as labeled data, and all the remaining data as unlabeled data.
        self.train_dataset, self.unsupervised_train_dataset = \
            random_split(self.train_dataset, [supervised_size, unsupervised_size])
        # cnn_dataset_to_jsonl("./data/train_100.jsonl", self.train_dataset)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def unsupervised_dataloader(self):
        assert getattr(self,
                       "unsupervised_train_dataset") is not None, "Unsupervised training dataset is not properly loaded!"

        return DataLoader(
            dataset=self.unsupervised_train_dataset + self.train_dataset,
            batch_size=self.train_batch_size * 8,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


# DataModule used for the NLU classification task, responsible for generating
# the dataloaders required during the training and evaluation phases.
class SelfUdaDataModule:
    def __init__(
            self,
            dataset_name: str,
            task_name: str,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            tokenizer_name_or_path: str,
            n_labeled_per_class: int,
            num_workers: int = 0,
    ):
        self.dataset_name = os.path.join("./data", dataset_name, "train.jsonl")

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
        self.task_name = task_name

        self.collator = udacollator

        train_filename = os.path.join("./data", dataset_name, "train_labeled_" + str(n_labeled_per_class) + ".jsonl")
        unlabeled_filename = os.path.join("./data", dataset_name, "train_unlabeled.jsonl")
        val_filename = os.path.join("./data", dataset_name, "val.jsonl")
        test_filename = os.path.join("./data", dataset_name, "test.jsonl")

        max_seq_len = 256
        self.train_labeled_dataset = SelfUdaDataset(data_filename=train_filename,
                                                    tokenizer=self.tokenizer,
                                                    max_seq_len=max_seq_len)
        self.train_unlabeled_dataset = SelfUdaDataset(data_filename=unlabeled_filename,
                                                      tokenizer=self.tokenizer,
                                                      max_seq_len=max_seq_len, )
        self.val_dataset = SelfUdaDataset(data_filename=val_filename,
                                          tokenizer=self.tokenizer,
                                          max_seq_len=max_seq_len)
        self.test_dataset = SelfUdaDataset(data_filename=test_filename,
                                           tokenizer=self.tokenizer,
                                           max_seq_len=max_seq_len)
        self.num_labels = len(self.train_labeled_dataset) // n_labeled_per_class
        log.info("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
            self.train_labeled_dataset), len(self.train_unlabeled_dataset), len(self.val_dataset),
            len(self.test_dataset)))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_labeled_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def unsupervised_dataloader(self):
        return DataLoader(
            dataset=self.train_unlabeled_dataset,
            batch_size=self.train_batch_size * 4,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


# Dataset class used for extractive summarization tasks.
class ConsistSumDataset(Dataset):
    def __init__(self,
                 data_filename,
                 tokenizer,
                 text_max_len: int,
                 max_text_ntokens_per_sent: int,
                 min_text_ntokens_per_sent: int,
                 ):
        self.data = read_jsonl(data_filename)

        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.max_text_ntokens_per_sent = max_text_ntokens_per_sent
        self.min_text_ntokens_per_sent = min_text_ntokens_per_sent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = self.data[index]
        text, summary, labels = ex["src"], ex["tgt"], ex["labels"]

        # Prepend [CLS] to the beginning of each sentence.
        add_special_token_text = list(map(lambda sent: self.tokenizer.cls_token + sent, text))
        subtokens = list(map(self.tokenizer.tokenize, add_special_token_text))

        # Remove excessively short sentences from the document.
        mask_idxs = [i for i, t in enumerate(subtokens) if len(t) > self.min_text_ntokens_per_sent]
        subtokens = [subtokens[idx] for idx in mask_idxs]
        text = [text[idx] for idx in mask_idxs]
        labels = [labels[idx] for idx in mask_idxs]

        # Truncate overly long sentences and append [SEP] at the end of all sentences.
        sent_ids = list(map(self.tokenizer.convert_tokens_to_ids, subtokens))
        sent_ids = [ids[:self.max_text_ntokens_per_sent - 1] + [self.tokenizer.sep_token_id] for ids in sent_ids]

        # Concatenate all sentences to form a complete document and truncate overly long documents.
        text_ids = list()
        for i in range(len(sent_ids)):
            if len(text_ids) + len(sent_ids[i]) <= self.text_max_len:
                text_ids.extend(sent_ids[i])
            else:
                remain_len = self.text_max_len - len(text_ids)
                if remain_len > self.min_text_ntokens_per_sent:
                    text_ids.extend(sent_ids[i][:remain_len - 1] + [self.tokenizer.sep_token_id])
                break

        text_ids = torch.tensor(text_ids)

        sent_num = (text_ids == self.tokenizer.cls_token_id).sum()

        sent_ids = sent_ids[:sent_num]
        labels = labels[:sent_num]

        return text_ids, labels, text, summary


# The dataset class used by TinyBERT in the General Distill phase.
class DistillDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        self.reduce_memory = reduce_memory
        log.info('training_path: {}'.format(training_path))
        self.data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        log.info('data_file: {}'.format(self.data_file))
        log.info('metrics_file: {}'.format(metrics_file))

        assert self.data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        self.num_samples = metrics['num_training_examples']
        self.seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None

    def generate_data(self):
        input_ids = np.zeros(shape=(self.num_samples, self.seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(self.num_samples, self.seq_len), dtype=np.bool)
        segment_ids = np.zeros(shape=(self.num_samples, self.seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(self.num_samples, self.seq_len), dtype=np.int32, fill_value=-1)
        is_nexts = np.zeros(shape=(self.num_samples,), dtype=np.bool)

        logging.info("Loading training examples for epoch {}".format(self.epoch))

        with self.data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, self.tokenizer, self.seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))


# The dataset class used for NLU classification tasks.
class SelfUdaDataset(Dataset):
    def __init__(self, data_filename, tokenizer, max_seq_len):
        lines = read_jsonl(data_filename)
        selfudadata = []
        n_labels = 0
        for (i, line) in enumerate(lines):
            selfudadata.append(
                UDAData(text=line["text"], label=line["label"]))
            if line["label"] > n_labels:
                n_labels = line["label"]
        self.data = convert_examples_to_features(selfudadata, range(n_labels + 1), max_seq_len, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =========================================================COLLATOR=============================================================


class Batch:
    def __init__(
            self,
            input_ids,
            attn_mask,
            pos_ids,
            cls_mask,
            sep_mask,
            seg,
            labels,
            texts,
            summaries,
    ):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.pos_ids = pos_ids
        self.cls_mask = cls_mask
        self.sep_mask = sep_mask
        self.seg = seg
        self.labels = labels
        self.texts = texts
        self.summaries = summaries

    def to(self, device):
        self.input_ids = self.input_ids.to(device) if self.input_ids is not None else None
        self.attn_mask = self.attn_mask.to(device) if self.attn_mask is not None else None
        self.pos_ids = self.pos_ids.to(device) if self.pos_ids is not None else None
        self.cls_mask = self.cls_mask.to(device) if self.cls_mask is not None else None
        self.sep_mask = self.sep_mask.to(device) if self.sep_mask is not None else None
        self.seg = self.seg.to(device) if self.seg is not None else None
        self.labels = self.labels.to(device) if self.labels is not None else None

        return self

    def __len__(self):
        return self.input_ids.size(0)


def pad_1d(x, pad_len, pad_id):
    xlen = x.size(0)
    if xlen < pad_len:
        new_x = x.new_empty([pad_len], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen] = x
        x = new_x
    elif xlen > pad_len:
        end_id = x[-1]
        x = x[:pad_len]
        x[-1] = end_id
    return x


def pad_2d(x, pad_len, pad_id):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < pad_len:
        new_x = x.new_zeros([pad_len, xdim], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen, :] = x
        x = new_x
    return x


# The collator used for the dataloader of extractive summarization tasks.
def collator(items, cls_id, sep_id, pad_id, text_max_len):
    input_ids, labels, texts, summaries = zip(*items)

    input_ids = torch.stack([pad_1d(ids, text_max_len, pad_id) for ids in input_ids], dim=0)
    attn_mask = ~(input_ids == pad_id)
    position_ids = torch.arange(input_ids.size(-1), dtype=torch.long).expand((1, -1))

    cls_mask = input_ids == cls_id
    sep_mask = input_ids == sep_id

    # Implement Interval Segment Embeddings as described in BertSum.
    # Assign alternating segment IDs of 0 or 1 based on the odd or even positions of sentences.
    tmp_input_ids = input_ids.view(-1, input_ids.size()[-1])
    segments_ids = []
    for content in tmp_input_ids:
        tmp = []
        _segs = [-1] + [i for i, t in enumerate(content) if t == sep_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                tmp += s * [0]
            else:
                tmp += s * [1]
        if len(tmp) < len(content):
            tmp += (len(content) - len(tmp)) * [0]
        segments_ids.append(tmp)
    segments_ids = torch.tensor(segments_ids).view(input_ids.size())

    labels = torch.tensor(sum(labels, list()))

    return Batch(
        input_ids=input_ids,
        attn_mask=attn_mask,
        pos_ids=position_ids,
        cls_mask=cls_mask,
        sep_mask=sep_mask,
        seg=segments_ids,
        labels=labels,
        texts=texts,
        summaries=summaries,
    )

# The collator used for the dataloader of NLU classification tasks.
def udacollator(items):
    input_ids, input_mask, segment_ids, label, texts = zip(*items)

    input_ids = torch.stack(input_ids, dim=0)
    input_mask = torch.stack(input_mask, dim=0)
    segment_ids = torch.stack(segment_ids, dim=0)
    label = torch.stack(label, dim=0)
    position_ids = torch.arange(input_ids.size(-1), dtype=torch.long).expand((1, -1))

    return Batch(
        input_ids=input_ids,
        attn_mask=input_mask,
        pos_ids=position_ids,
        cls_mask=None,
        sep_mask=None,
        seg=segment_ids,
        labels=label,
        texts=texts,
        summaries=None,
    )
