import os
import time
import traceback
import logging
import shutil
import re
import string
from collections import Counter
from sklearn.metrics import f1_score

from pyrouge import Rouge155
import rouge

from utils import get_logger

log = get_logger(__name__)


def calc_rouge_from_pyrouge(hypothesis, references):
    now_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tmp_dir = f"./tmp/rouge_tmp_f{now_time}"

    os.makedirs(os.path.join(tmp_dir, "hypothesis"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "references"), exist_ok=True)

    try:
        for idx, (hyp, ref) in enumerate(zip(hypothesis, references)):
            if len(ref) < 1:
                continue
            with open(os.path.join(tmp_dir, "hypothesis", f"hyp.{idx}.txt"), "w") as f:
                f.write(hyp)
            with open(os.path.join(tmp_dir, "references", f"ref.{idx}.txt"), "w") as f:
                f.write(ref)

        r = Rouge155(log_level=logging.ERROR)
        r.model_dir = os.path.join(tmp_dir, "references")
        r.system_dir = os.path.join(tmp_dir, "hypothesis")
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"hyp.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        dict_scores = r.output_to_dict(rouge_results)

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())

    finally:
        shutil.rmtree(tmp_dir)

    scores = {
        "rouge1_F1": round(dict_scores["rouge_1_f_score"], 4) * 100.0,
        "rouge2_F1": round(dict_scores["rouge_2_f_score"], 4) * 100.0,
        "rougel_F1": round(dict_scores["rouge_l_f_score"], 4) * 100.0,
        "rouge1_R": round(dict_scores["rouge_1_recall"], 4) * 100.0,
        "rouge2_R": round(dict_scores["rouge_2_recall"], 4) * 100.0,
        "rougel_R": round(dict_scores["rouge_l_recall"], 4) * 100.0,
        "rouge1_P": round(dict_scores["rouge_1_precision"], 4) * 100.0,
        "rouge2_P": round(dict_scores["rouge_2_precision"], 4) * 100.0,
        "rougel_P": round(dict_scores["rouge_l_precision"], 4) * 100.0
    }

    return scores


def calc_rouge_from_python_implementation(hypothesis, references):
    criterion = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2)
    rouge_results = criterion.get_scores(hypothesis, references)

    scores = {
        "rouge1": round(rouge_results["rouge-1"]["f"], 4) * 100.0,
        "rouge2": round(rouge_results["rouge-2"]["f"], 4) * 100.0,
        "rougel": round(rouge_results["rouge-l"]["f"], 4) * 100.0
    }

    return scores


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())
