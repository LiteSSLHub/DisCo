import os
import traceback
from argparse import ArgumentParser
import random
import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import linalg as LA
import torch.nn.functional as F
from transformers.utils.logging import set_verbosity_error

from evaluate import evaluate
from model import ConsistSum
from data import ConsistSumDataModule, SelfUdaDataModule
from utils import (
    load_checkpoints,
    save_checkpoints,
    get_logger
)

log = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    # training arguments
    subparser = parser.add_argument_group("train")
    subparser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    subparser.add_argument("--cuda", action="store_true", help="GPU or CPU.")
    subparser.add_argument("--seed", type=int, default=42, help="Random seed.")
    subparser.add_argument("--root_dir", type=str, default="./experiments/cnndm",
                           help="The root directory of this run.")
    subparser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                           help="The specific directory name in the root directory to save checkpoints.")
    subparser.add_argument("--test_after_train", action="store_true", default=True, help="Do test after training.")
    subparser.add_argument("--do_consist", action="store_true", help="Consistency training or not.")
    subparser.add_argument("--task_name", type=str, default="sum",
                           choices=["sum", "yahoo", "dbpedia", "agnews"],
                           help="specific task name.")
    subparser.add_argument("--supervised_size", type=int, default=50000,
                           help="Number of supervised data in consistency training.")
    subparser.add_argument("--unsupervised_size", type=int, default=50000,
                           help="Number of unsupervised data in consistency training.")
    subparser.add_argument("--lr", type=float, default=2e-3, help="Base learning rate.")
    subparser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay of adamW.")
    subparser.add_argument("--num_training_steps", type=int, default=50000, help="Total number of training steps.")
    subparser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    subparser.add_argument("--warmup_proportion", default=0.2, type=float,
                           help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    subparser.add_argument("--val_interval", type=int, default=2000, help="Intervals for evaluate.")
    subparser.add_argument("--resume_ckpt_path", type=str, default=None, help="Resume checkpoint path.")
    subparser.add_argument("--do_adversary", action="store_true", help="generate adversary data or not.")
    subparser.add_argument("--do_block", action="store_true", default=True, help="Trigram block or not.")
    subparser.add_argument("--lambdau", type=float, default=100,
                           help="Hyperparameters representing the importance of mutual learning.")
    subparser.add_argument("--rampup_rate", type=float, default=0.2,
                           help="Proportion of training to perform ramp up step.")
    subparser.add_argument("--student_num", default=2, type=int, required=True, help="Number of student model.")
    subparser.add_argument("--data_difference", action="store_true", help="make supervised data the same or not.")
    subparser.add_argument("--do_aug", type=int, default=0,
                           help="Make data augmentation method to the inputs of the two models.")
    subparser.add_argument("--cutoff_rate", type=float, default=0.2, help="The final cutoff/dropout rate")

    # dataset arguments
    subparser = parser.add_argument_group("data")
    subparser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset.")
    subparser.add_argument("--train_batch_size", type=int, default=32, help="Batch size of training.")
    subparser.add_argument("--val_batch_size", type=int, default=32, help="Batch size of validation.")
    subparser.add_argument("--test_batch_size", type=int, default=32, help="Batch size of testing.")
    subparser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased",
                           # choices=["bert-base-uncased", "bert-large-uncased"],
                           help="The name or path of pretrained tokenizer.")
    subparser.add_argument("--num_workers", type=int, default=8, help="Number of process workers in dataloader.")
    subparser.add_argument("--extract_nsents", type=int, default=3, help="Number of oracle summary.")

    # model arguments
    subparser = parser.add_argument_group("model")
    subparser.add_argument("--encoder_name_or_path", type=str, default="bert-base-uncased",
                           # choices=["bert-base-uncased", "bert-large-uncased"],
                           help="The name or path of pretrained language model.")
    subparser.add_argument("--config_path", type=str, default=None,
                           help="The path of the config file.")
    subparser.add_argument("--ensemble_prediction", action="store_true",
                           help="Use all students or use the first student to predict.")
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Linear ramp-up function for the consistency loss section, starts at 0 at the beginning
# of training and linearly increases until it reaches 1 at 'ramp_up_rate * total_step.
def linear_rampup(current, args):
    rampup_length = args.num_training_steps * args.rampup_rate
    current = np.clip(current / rampup_length, 0.0, 1.0)
    return float(current)


# Used to expand 'input_ids' and 'input_mask' inputs for data augmentation in 'cut_off' and 'shuffle.
def trans_view(input_tensor):
    return input_tensor.view(-1, input_tensor.size(-1))


# Generate adversarial augmentation 'input_embeds'.
def vat_generator(model, x, specific_student, inputs_embeds, target_logits, iter, device=None, loss_func=None,
                  adv_step_size=1e-3, adv_epsilon=1e-6, adv_noise_var=1e-5):
    inputs_embeds = inputs_embeds.detach()
    target_logits = target_logits.detach()
    noise = (inputs_embeds.data.new(inputs_embeds.size()).normal_(0, 1) * adv_noise_var).to(device)
    noise.detach()
    noise.requires_grad_()
    # If 'loss_func' is provided, it's labeled data, use corresponding 'CE_loss' or 'BCE_loss';
    # otherwise, it's unlabeled data, use 'MSE_loss.'
    loss = nn.MSELoss(reduction='sum') if loss_func is None else loss_func
    for step in range(iter):
        adv_logits, _ = model(batch=x,
                              specific_student=specific_student,
                              inputs_embeds=inputs_embeds + noise)
        adv_loss = loss(adv_logits[0], target_logits)
        # Compute gradients of the loss with respect to noise.
        delta_grad, = torch.autograd.grad(adv_loss, noise, retain_graph=False)
        norm = delta_grad.norm()
        if torch.isnan(norm) or torch.isinf(norm):
            return inputs_embeds
        # Backward gradient descent to make noise move in the direction that maximizes the loss.
        delta_grad = noise + delta_grad * adv_step_size
        noise = delta_grad / (LA.norm(delta_grad, dim=-1, keepdim=True) + adv_epsilon)
        noise = noise.detach()
        noise.requires_grad_()
    return inputs_embeds + noise


# Similar to the 'shuffle' function but keeps the position of [CLS] tokens unchanged
# and shuffles only the position IDs of tokens within each [CLS].
def local_shuffle(batch):
    tmp_batch = copy.deepcopy(batch)
    input_ids, attn_mask = trans_view(tmp_batch.input_ids), trans_view(tmp_batch.attn_mask)
    bsz, seq_len = input_ids.shape
    position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len]
    # shuffle position_ids
    shuffled_pid = []
    for bsz_id in range(bsz):
        sample_pid = position_ids[bsz_id]
        sample_mask = attn_mask[bsz_id]
        num_tokens = sample_mask.sum().int().item()
        # Get the position of [CLS] tokens.
        cls_pos = [i for i, t in enumerate(input_ids[bsz_id]) if t == 101] + [num_tokens]
        tmp_shuffled_pid = []
        for i in range(len(cls_pos) - 1):
            # Get the position of [CLS] tokens.
            tmp_index = list(range(cls_pos[i] + 1, cls_pos[i + 1]))
            random.shuffle(tmp_index)
            tmp_shuffled_pid += [cls_pos[i]] + tmp_index
        rest_indexes = list(range(num_tokens, seq_len))
        total_indexes = tmp_shuffled_pid + rest_indexes
        shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes)))
    tmp_batch.pos_ids = torch.stack(shuffled_pid, 0).to(input_ids.device)
    return tmp_batch


# Enhanced dropout uses the one built into the BERT model, so simply return the original batch.
def dropout(batch):
    return batch


# Cut-off data augmentation, set 'attention_mask' to 0 for a random 'cut_off_rate * num_tokens' number of tokens
# in the input tokens. Ensure that the 'attention_mask' corresponding to [CLS] is always 1.
def cutoff(batch, rate):
    tmp_batch = copy.deepcopy(batch)
    input_ids, attn_mask = trans_view(tmp_batch.input_ids), trans_view(tmp_batch.attn_mask)
    bsz, seq_len = input_ids.shape
    cutoff_pid = []
    for bsz_id in range(bsz):
        num_tokens = attn_mask[bsz_id].sum().int().item()
        num_cutoff_indexes = int(num_tokens * rate)
        if num_cutoff_indexes < 0 or num_cutoff_indexes > num_tokens:
            raise ValueError(
                f"number of cutoff dimensions should be in (0, {num_tokens}), but got {num_cutoff_indexes}")
        indexes = list(range(num_tokens))
        random.shuffle(indexes)
        cutoff_indexes = indexes[:num_cutoff_indexes]
        # 101 is the token ID corresponding to [CLS] in BERT.
        cutoff_pid.append(torch.tensor(
            [attn_mask[bsz_id][i] if ((input_ids[bsz_id][i] == 101) or (i not in cutoff_indexes)) else 0 for i in
             range(seq_len)]))
    tmp_batch.attn_mask = torch.stack(cutoff_pid, 0).to(input_ids.device)
    return tmp_batch


# Invoke the 'vat_generator' function and provide different loss functions and labels to
# 'vat_generator' based on the task and data type.
def adversary(batch, model, student_num, device, label=None, task_name=None):
    if task_name == "sum":
        loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_func = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        logits, student_embeds = model(batch, do_adversary=True, specific_student=student_num)
    adversary_embeds = vat_generator(model, batch, student_num, student_embeds[0],
                                     logits[0] if label is None else label, 1,
                                     device, None if label is None else loss_func)
    return adversary_embeds


def get_optimizer_and_scheduler(model, args):
    def lr_lambda(current_step):
        current_step = current_step + 1
        return min(current_step ** -0.5, current_step * (
                (args.num_training_steps / args.gradient_accumulation_steps * args.warmup_proportion) ** -1.5))

    def get_decay_parameter(submodel):
        # Prepare optimizer
        param_optimizer = list(submodel.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    optimizer = []
    scheduler = []
    for n, p in model.named_children():
        if n == "student":
            student = p
        elif n == "head":
            head = p

    student_list = list(student.children())
    head_list = list(head.children())
    for i in range(args.student_num):
        optimizer.append(torch.optim.AdamW(
            get_decay_parameter(student_list[i]) + get_decay_parameter(head_list[i]),
            lr=args.lr))
        scheduler.append(torch.optim.lr_scheduler.LambdaLR(optimizer[i], lr_lambda))
    return optimizer, scheduler


# Return the data augmentation method for each student based on the 'do_aug' parameter.
# This function is only used for the case of two students.
def get_aug_method(args):
    all_method = [local_shuffle, partial(cutoff, rate=args.cutoff_rate), dropout, adversary]
    all_method_name = ["shuffle", "cutoff", "dropout", "adversary"]
    return [all_method[(args.do_aug - 1) % 4], all_method[(args.do_aug - 1) // 4]], [
        all_method_name[(args.do_aug - 1) % 4], all_method_name[(args.do_aug - 1) // 4]]


# Main function for student training.
def consist_train(args):
    log.info(f'args:{args}')

    if args.task_name == "sum":
        datamodule = ConsistSumDataModule(dataset_name=args.dataset_name,
                                          train_batch_size=args.train_batch_size,
                                          val_batch_size=args.val_batch_size,
                                          test_batch_size=args.test_batch_size,
                                          tokenizer_name_or_path=args.tokenizer_name_or_path,
                                          num_workers=args.num_workers)
        datamodule.prepare(supervised_dataset_size=args.supervised_size)
        num_labels = 1
    elif args.task_name in ["yahoo",  "dbpedia", "agnews"]:
        datamodule = SelfUdaDataModule(dataset_name=args.dataset_name,
                                       task_name=args.task_name,
                                       train_batch_size=args.train_batch_size,
                                       val_batch_size=args.val_batch_size,
                                       test_batch_size=args.test_batch_size,
                                       tokenizer_name_or_path=args.tokenizer_name_or_path,
                                       n_labeled_per_class=args.supervised_size,
                                       num_workers=args.num_workers)
        num_labels = datamodule.num_labels

    model = ConsistSum(args.encoder_name_or_path,
                       student_num=args.student_num,
                       num_classes=num_labels,
                       task_name=args.task_name,
                       aug_num=args.do_aug)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    if args.do_aug != 0:
        aug_method, aug_method_name = get_aug_method(args)
        log.info(f"model_1 uses {aug_method_name[0]}, model_2 uses {aug_method_name[1]}")

    # Here, a dataloader is constructed for each student to experiment with different students training
    # on the same training set in different orders to further promote differences between models.
    # However, the final results indicate that this approach has little utility.
    train_dataloader = [datamodule.train_dataloader() for _ in range(args.student_num)]
    unsupervised_dataloader = datamodule.unsupervised_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = [iter(dl) for dl in train_dataloader]
    unsupervised_dataiter = iter(unsupervised_dataloader)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        model = model.to(torch.device(args.gpu))

    if args.task_name == "sum":
        loss_ce = nn.BCEWithLogitsLoss()
    else:
        loss_ce = nn.CrossEntropyLoss()

    current_step = 0
    train_loss = [0.0] * args.student_num
    supervised_loss = [0.0] * args.student_num
    unsupervised_loss = [0.0] * args.student_num
    raw_mutual_loss = [0.0] * args.student_num
    raw_entropy_regularization_loss = [0.0] * args.student_num
    best_eval_loss = [np.inf] * args.student_num
    best_rouge_score = [np.NINF] * args.student_num
    best_loss_checkpoints_filename = [""] * args.student_num
    best_rouge_checkpoints_filename = [""] * args.student_num

    if args.resume_ckpt_path is not None:
        log.info(f"Resume from {args.resume_ckpt_path}...")
        ckpt = load_checkpoints(args.resume_ckpt_path, args.gpu if args.cuda else "cpu")
        current_step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])

    log.info("***** Running training *****")
    log.info("  Supervised Batch size = %d", train_dataloader[0].batch_size)
    log.info("  Unsupervised Batch size = %d", unsupervised_dataloader.batch_size)
    log.info("  Num steps = %d", args.num_training_steps)
    while current_step < args.num_training_steps:
        model.train()

        try:
            batch = [next(diter) for diter in train_dataiter]
        except StopIteration:
            train_dataiter = [iter(dl) for dl in train_dataloader]
            batch = [next(diter) for diter in train_dataiter]
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        try:
            unsupervised_batch = next(unsupervised_dataiter)
        except StopIteration:
            unsupervised_dataiter = iter(unsupervised_dataloader)
            unsupervised_batch = next(unsupervised_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        batch, unsupervised_batch = [content.to(device) for content in batch], unsupervised_batch.to(device)
        loss_mse = nn.MSELoss()
        all_unsupervised_logits = []
        # Calculate the cross-entropy loss for each student on labeled data and
        # save the results for each student on unlabeled data.
        for bias in range(args.student_num):
            # If not experimenting with 'different students training on the same training set in different orders,
            # ' all students can use the batch output by the first dataloader.
            specific_batch = batch[bias] if args.data_difference else batch[0]
            supervised_input_embeds = None
            unsupervised_input_embeds = None
            # Backup and retain the values of the original batch so that different students can generate
            # different augmented data starting from the original batch.
            tmp_specific_batch = specific_batch
            tmp_unsupervised_batch = unsupervised_batch
            if args.do_aug != 0:
                if aug_method[bias % 2] is adversary:
                    supervised_input_embeds = adversary(specific_batch, model, bias, device, specific_batch.labels,
                                                        task_name=args.task_name)
                    unsupervised_input_embeds = adversary(unsupervised_batch, model, bias, device,
                                                          task_name=args.task_name)
                else:
                    tmp_specific_batch = aug_method[bias % 2](specific_batch)
                    tmp_unsupervised_batch = aug_method[bias % 2](unsupervised_batch)

            logits, _ = model(tmp_specific_batch, inputs_embeds=supervised_input_embeds, specific_student=bias)
            unsupervised_logits, _ = model(tmp_unsupervised_batch, inputs_embeds=unsupervised_input_embeds,
                                           specific_student=bias)

            ce_loss = loss_ce(logits[0], tmp_specific_batch.labels) / args.gradient_accumulation_steps
            # Save the results of running unlabeled data for the model. Calculate mutual learning consistency loss
            # once results for all students are obtained.
            all_unsupervised_logits.append(unsupervised_logits[0])
            ce_loss.backward()
            train_loss[bias] += ce_loss.item() * args.gradient_accumulation_steps
            supervised_loss[bias] += ce_loss.item() * args.gradient_accumulation_steps

        # Calculate the mutual learning consistency loss among all students on unlabeled data.
        for bias in range(args.student_num):
            mutual_loss = 0.0
            for other_student in range(args.student_num):
                if other_student != bias:
                    mutual_loss += loss_mse(all_unsupervised_logits[bias],
                                            all_unsupervised_logits[other_student].detach())
            weighted_mutual_loss = args.lambdau * linear_rampup(current_step, args) * mutual_loss / (
                    args.student_num - 1) / args.gradient_accumulation_steps if args.student_num > 1 else 0

            if args.student_num > 1:
                weighted_mutual_loss.backward()
            if (current_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer[bias].step()
                scheduler[bias].step()
                optimizer[bias].zero_grad()

            train_loss[bias] += weighted_mutual_loss.item() * args.gradient_accumulation_steps if args.student_num > 1 else 0
            raw_mutual_loss[bias] += mutual_loss.item() if args.student_num > 1 else 0.0
            unsupervised_loss[bias] += weighted_mutual_loss.item() * args.gradient_accumulation_steps if args.student_num > 1 else 0

        current_step = current_step + 1

        # Track training loss.
        if current_step % 100 == 0:
            for bias in range(args.student_num):
                log.info(
                    f"Student {bias} in Step {current_step:3d} | train loss {(train_loss[bias] / 100.0):5.4f} | supervised loss {(supervised_loss[bias] / 100.0):5.4f} | raw mutual loss {(raw_mutual_loss[bias] / 100.0):5.4f} | entropy regularization loss {(raw_entropy_regularization_loss[bias] / 100.0):5.4f} | unsupervised loss {(unsupervised_loss[bias] / 100.0):5.4f}")
                train_loss[bias] = 0.0
                supervised_loss[bias] = 0.0
                unsupervised_loss[bias] = 0.0
                raw_mutual_loss[bias] = 0.0
                raw_entropy_regularization_loss[bias] = 0.0

        # Dev Set Evaluation
        if current_step % args.val_interval == 0:
            for bias in range(args.student_num):
                eval_loss, rouge_scores = evaluate(model,
                                                   val_dataloader,
                                                   bias,
                                                   args.extract_nsents,
                                                   device,
                                                   num_labels=num_labels,
                                                   task_name=args.task_name,
                                                   pyrouge=False,
                                                   trigram_block=args.do_block,
                                                   ensemble_prediction=args.ensemble_prediction)

                checkpoints = {
                    "step": current_step,
                    "model": model.state_dict()
                }
                checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
                save_checkpoints(checkpoints_filename, checkpoints)

                # Save the model parameters with the lowest loss and highest evaluation metrics on the dev set.
                if eval_loss < best_eval_loss[bias]:
                    best_eval_loss[bias] = eval_loss
                    best_loss_checkpoints_filename[bias] = checkpoints_filename
                tmp_rouge_score = sum(rouge_scores.values())
                if tmp_rouge_score > best_rouge_score[bias]:
                    best_rouge_score[bias] = tmp_rouge_score
                    best_rouge_checkpoints_filename[bias] = checkpoints_filename

    log.info("Train end.")

    # Test Set Evaluation
    if args.test_after_train:
        for bias in range(args.student_num):
            log.info(f"For student {bias}, the best loss checkpoint file is in {best_loss_checkpoints_filename[bias]}")
            log.info(
                f"For student {bias}, the best rouge checkpoint file is in {best_rouge_checkpoints_filename[bias]}")
            ckpt = load_checkpoints(best_loss_checkpoints_filename[bias], device)
            model.load_state_dict(ckpt["model"])
            log.info("Test the best loss checkpoints.")
            evaluate(model,
                     test_dataloader,
                     bias,
                     args.extract_nsents,
                     device,
                     num_labels=num_labels,
                     task_name=args.task_name,
                     pyrouge=True,
                     trigram_block=args.do_block,
                     ensemble_prediction=args.ensemble_prediction)
            ckpt = load_checkpoints(best_rouge_checkpoints_filename[bias], device)
            model.load_state_dict(ckpt["model"])
            log.info("Test the best rouge checkpoints.")
            evaluate(model,
                     test_dataloader,
                     bias,
                     args.extract_nsents,
                     device,
                     num_labels=num_labels,
                     task_name=args.task_name,
                     pyrouge=True,
                     trigram_block=args.do_block,
                     ensemble_prediction=args.ensemble_prediction)

    return best_loss_checkpoints_filename, best_rouge_checkpoints_filename


# Similar to the 'consist_train' function, but 'train' function is specifically designed for training a single student,
# so it does not require unlabeled data and does not calculate consistency loss.
def train(args):
    args.student_num = 1
    log.info(f'args:{args}')

    if args.task_name == "sum":
        datamodule = ConsistSumDataModule(dataset_name=args.dataset_name,
                                          train_batch_size=args.train_batch_size,
                                          val_batch_size=args.val_batch_size,
                                          test_batch_size=args.test_batch_size,
                                          tokenizer_name_or_path=args.tokenizer_name_or_path,
                                          num_workers=args.num_workers)
        datamodule.prepare(supervised_dataset_size=args.supervised_size)
        num_labels = 1
    elif args.task_name in ["yelp-2", "yelp-5", "yahoo", "imdb", "dbpedia", "agnews"]:
        datamodule = SelfUdaDataModule(dataset_name=args.dataset_name,
                                       task_name=args.task_name,
                                       train_batch_size=args.train_batch_size,
                                       val_batch_size=args.val_batch_size,
                                       test_batch_size=args.test_batch_size,
                                       tokenizer_name_or_path=args.tokenizer_name_or_path,
                                       n_labeled_per_class=args.supervised_size,
                                       num_workers=args.num_workers)
        num_labels = datamodule.num_labels

    model = ConsistSum(args.encoder_name_or_path,
                       student_num=args.student_num,
                       num_classes=num_labels,
                       task_name=args.task_name,
                       aug_num=args.do_aug)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = iter(train_dataloader)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        model = model.to(torch.device(args.gpu))

    if args.task_name == "sum":
        loss_ce = nn.BCEWithLogitsLoss()
    else:
        loss_ce = nn.CrossEntropyLoss()

    current_step = 0
    train_loss = 0.0
    best_eval_loss = np.inf
    best_rouge_score = np.NINF
    best_loss_checkpoints_filename = None
    best_rouge_checkpoints_filename = None
    if args.resume_ckpt_path is not None:
        log.info(f"Resume from {args.resume_ckpt_path}...")
        ckpt = load_checkpoints(args.resume_ckpt_path, args.gpu if args.cuda else "cpu")
        current_step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])

    log.info("***** Running training *****")
    log.info("  Supervised Batch size = %d", train_dataloader.batch_size)
    log.info("  Num steps = %d", args.num_training_steps)
    while current_step < args.num_training_steps:
        model.train()

        try:
            batch = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")

        batch = batch.to(device)
        logits, _ = model(batch)

        ce_loss = loss_ce(logits[0], batch.labels) / args.gradient_accumulation_steps
        loss = ce_loss
        loss.backward()

        if (current_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer[0].step()
            scheduler[0].step()
            optimizer[0].zero_grad()

        current_step = current_step + 1
        train_loss += loss.data.item() * args.gradient_accumulation_steps

        if current_step % 100 == 0:
            log.info(f"Step {current_step:3d} | train loss {(train_loss / 100.0):5.4f}")
            train_loss = 0.0

        if current_step % args.val_interval == 0:
            eval_loss, rouge_scores = evaluate(model,
                                               val_dataloader,
                                               0,
                                               args.extract_nsents,
                                               device,
                                               num_labels=num_labels,
                                               task_name=args.task_name,
                                               pyrouge=False,
                                               trigram_block=args.do_block,
                                               ensemble_prediction=args.ensemble_prediction)

            checkpoints = {
                "step": current_step,
                "model": model.state_dict()
            }
            checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
            save_checkpoints(checkpoints_filename, checkpoints)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_loss_checkpoints_filename = checkpoints_filename
            tmp_rouge_score = sum(rouge_scores.values())
            if tmp_rouge_score > best_rouge_score:
                best_rouge_score = tmp_rouge_score
                best_rouge_checkpoints_filename = checkpoints_filename

    log.info("Train end.")
    log.info(f"The best loss checkpoint file is in {best_loss_checkpoints_filename}")
    log.info(f"The best rouge checkpoint file is in {best_rouge_checkpoints_filename}")

    if args.test_after_train:
        ckpt = load_checkpoints(best_loss_checkpoints_filename, device)
        model.load_state_dict(ckpt["model"])
        log.info("Test the best loss checkpoints.")
        evaluate(model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 num_labels=num_labels,
                 task_name=args.task_name,
                 pyrouge=True,
                 trigram_block=args.do_block,
                 ensemble_prediction=args.ensemble_prediction)
        ckpt = load_checkpoints(best_rouge_checkpoints_filename, device)
        model.load_state_dict(ckpt["model"])
        log.info("Test the best rouge checkpoints.")
        evaluate(model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 num_labels=num_labels,
                 task_name=args.task_name,
                 pyrouge=True,
                 trigram_block=args.do_block,
                 ensemble_prediction=args.ensemble_prediction)

    return best_loss_checkpoints_filename, best_rouge_checkpoints_filename


if __name__ == "__main__":
    set_verbosity_error()
    args = parse_args()

    if args.seed > 0:
        log.info(f"Set seed to {args.seed}")
        seed_everything(args.seed)

    if args.do_consist:
        log.info(f"Do consistency training!")
        best_loss_checkpoints_filename, best_rouge_checkpoints_filename = consist_train(args)
    else:
        best_loss_checkpoints_filename, best_rouge_checkpoints_filename = train(args)
