# coding=utf-8
# 2019.12.2-Changed for TinyBERT general distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from collections import namedtuple
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from model import TinyBertForPreTraining
from utils import WEIGHTS_NAME, CONFIG_NAME
from data import DistillDataset

csv.field_size_limit(sys.maxsize)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def get_optimizer_and_scheduler(model, args, num_train_optimization_steps):
    def get_decay_parameter(submodel):
        # Prepare optimizer
        param_optimizer = list(submodel.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    optimizer = []
    scheduler = []
    for n, p in model.named_children():
        logger.info(f"parameter name is {n}")
        if n == "student":
            student = p
        elif n == "fit_dense":
            fit_dense = p

    student_list = list(student.children())
    fit_dense_list = list(fit_dense.children())
    for i in range(args.student_num):
        optimizer.append(
            torch.optim.AdamW(get_decay_parameter(student_list[i]) + get_decay_parameter(fit_dense_list[i]),
                              lr=args.learning_rate))
        scheduler.append(get_linear_schedule_with_warmup(optimizer[i], num_warmup_steps=np.floor(
            num_train_optimization_steps * args.warmup_proportion),
                                                         num_training_steps=num_train_optimization_steps))
    return optimizer, scheduler


class Trainer(object):
    def __init__(self):
        self.samples_per_epoch = []
        for i in range(int(args.num_train_epochs)):
            epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
            metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
            if epoch_file.is_file() and metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text())
                self.samples_per_epoch.append(metrics['num_training_examples'])
            else:
                if i == 0:
                    exit("No training data was found!")
                print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i,
                                                                                                                    args.num_train_epochs))
                print(
                    "This script will loop over the available data, but training diversity may be negatively impacted.")
                self.num_data_epochs = i
                break
        else:
            self.num_data_epochs = args.num_train_epochs

        self.tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

        self.total_train_examples = 0
        for i in range(int(args.num_train_epochs)):
            # The modulo takes into account the fact that we may loop over limited epochs of data
            # That is, 'num_train_epochs' may be greater than the size of 'samples_per_epoch,' which can result in samples being repeatedly trained
            self.total_train_examples += self.samples_per_epoch[
                i % len(self.samples_per_epoch)]

        self.all_dataset = []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  # Progress bar.
            epoch_dataset = DistillDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=self.tokenizer,
                                           num_data_epochs=self.num_data_epochs, reduce_memory=args.reduce_memory)
            epoch_dataset.generate_data()
            self.all_dataset.append(epoch_dataset)

    def train(self, local_rank, args):
        if local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            rank = args.node_num * args.ngpus_per_node + local_rank
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://localhost:23456',
                                                 world_size=args.world_size,
                                                 rank=rank)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
        logger = logging.getLogger(__name__)

        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, args.world_size, bool(local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.world_size > 0 and not args.no_cuda:
            torch.cuda.manual_seed_all(args.seed)

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        num_train_optimization_steps = int(
            self.total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
        if local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // args.world_size

        teacher_model = BertModel.from_pretrained(args.teacher_model)
        if args.continue_train:
            student_model = TinyBertForPreTraining.from_pretrained(args.student_model)
        else:
            resolved_config_file = os.path.join(
                args.student_model, CONFIG_NAME)
            config = BertConfig.from_json_file(resolved_config_file)
            student_model = TinyBertForPreTraining(config, student_num=args.student_num,
                                                   fit_size=teacher_model.config.hidden_size)

        # student_model = TinyBertForPreTraining.from_scratch(args.student_model, fit_size=teacher_model.config.hidden_size)
        student_model.to(device)
        teacher_model.to(device)

        if local_rank != -1:
            teacher_model = DistributedDataParallel(teacher_model, device_ids=[local_rank], find_unused_parameters=True)
            student_model = DistributedDataParallel(student_model, device_ids=[local_rank], find_unused_parameters=True)

        size = 0
        for n, p in student_model.named_parameters():
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))

        # Prepare optimizer

        loss_mse = MSELoss()
        optimizer, scheduler = get_optimizer_and_scheduler(student_model, args, num_train_optimization_steps)

        global_step = 0
        logging.info("***** Running training *****")
        logging.info("  Num examples = {}".format(self.total_train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        for epoch in range(int(args.num_train_epochs)):  # 进度条
            epoch_dataset = self.all_dataset[epoch]
            if local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            # The purpose of setting the 'sampler' is to customize the sampling strategy for the dataset.
            # Here, setting 'RandomSampler' is equivalent to 'shuffle=True.'
            # Additionally, there is 'SequentialSampler,' which is equivalent to 'shuffle=False.
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            tr_loss = [0.0] * args.student_num
            tr_att_loss = [0.0] * args.student_num
            tr_rep_loss = [0.0] * args.student_num
            student_model.train()
            nb_tr_steps = 0
            with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                    batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                    if input_ids.size()[0] != args.train_batch_size:
                        continue

                    student_encoder_atts, student_encoder_layers = student_model(input_ids, segment_ids, input_mask,
                                                                                 output_hidden_states=True,
                                                                                 output_attentions=True)
                    with torch.no_grad():
                        # teacher_output=("last_hidden_state", "pooler_output", "all_hidden_states", "attentions")
                        teacher_output = teacher_model(input_ids, segment_ids, input_mask,
                                                       output_hidden_states=True, output_attentions=True)
                    teacher_encoder_atts = teacher_output.attentions
                    teacher_encoder_layers = teacher_output.hidden_states

                    teacher_layer_num = len(teacher_encoder_atts)
                    student_layer_num = len(student_encoder_atts[0])
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)

                    for bias in range(args.student_num):
                        if args.distill_mode == 0:
                            att_loss = 0.0
                            new_teacher_atts = [teacher_encoder_atts[i * layers_per_block + layers_per_block - 1 - bias]
                                                for i in range(student_layer_num - 1)] + [
                                                   teacher_encoder_atts[teacher_layer_num - 1]]
                            for student_att, teacher_att in zip(student_encoder_atts[bias], new_teacher_atts):
                                if args.att_zero_like:
                                    student_att = torch.where(student_att <= -1e2,
                                                              torch.zeros_like(student_att).to(device),
                                                              student_att)
                                    teacher_att = torch.where(teacher_att <= -1e2,
                                                              torch.zeros_like(teacher_att).to(device),
                                                              teacher_att)
                                att_loss += loss_mse(student_att, teacher_att)

                            rep_loss = 0.0
                            new_teacher_reps = [teacher_encoder_layers[0]] + \
                                               [teacher_encoder_layers[(i + 1) * layers_per_block - bias] for i in
                                                range(student_layer_num - 1)] + \
                                               [teacher_encoder_layers[teacher_layer_num]]
                            for student_rep, teacher_rep in zip(student_encoder_layers[bias], new_teacher_reps):
                                rep_loss += loss_mse(student_rep, teacher_rep)

                            loss = att_loss + rep_loss
                        if args.distill_mode == 1:
                            att_loss = 0.0
                            new_teacher_atts = [teacher_encoder_atts[bias * student_layer_num + i] for
                                                i in range(student_layer_num)]
                            for student_att, teacher_att in zip(student_encoder_atts[bias], new_teacher_atts):
                                if args.att_zero_like:
                                    student_att = torch.where(student_att <= -1e2,
                                                              torch.zeros_like(student_att).to(device),
                                                              student_att)
                                    teacher_att = torch.where(teacher_att <= -1e2,
                                                              torch.zeros_like(teacher_att).to(device),
                                                              teacher_att)
                                att_loss += loss_mse(student_att, teacher_att)

                            rep_loss = 0.0
                            new_teacher_reps = [teacher_encoder_layers[bias * student_layer_num + i] for i in
                                                range(student_layer_num + 1)]
                            for student_rep, teacher_rep in zip(student_encoder_layers[bias], new_teacher_reps):
                                rep_loss += loss_mse(student_rep, teacher_rep)

                            loss = att_loss + rep_loss
                        if args.distill_mode == 2:
                            att_loss = 0.0
                            new_teacher_atts = [teacher_encoder_atts[i * layers_per_block + layers_per_block - 1 - bias]
                                                for i in range(student_layer_num)]
                            for student_att, teacher_att in zip(student_encoder_atts[bias], new_teacher_atts):
                                if args.att_zero_like:
                                    student_att = torch.where(student_att <= -1e2,
                                                              torch.zeros_like(student_att).to(device),
                                                              student_att)
                                    teacher_att = torch.where(teacher_att <= -1e2,
                                                              torch.zeros_like(teacher_att).to(device),
                                                              teacher_att)
                                att_loss += loss_mse(student_att, teacher_att)

                            rep_loss = 0.0
                            new_teacher_reps = [teacher_encoder_layers[0]] + \
                                               [teacher_encoder_layers[(i + 1) * layers_per_block - bias] for i in
                                                range(student_layer_num)]
                            for student_rep, teacher_rep in zip(student_encoder_layers[bias], new_teacher_reps):
                                rep_loss += loss_mse(student_rep, teacher_rep)

                            loss = att_loss + rep_loss

                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()

                        tr_att_loss[bias] += att_loss.item()
                        tr_rep_loss[bias] += rep_loss.item()
                        tr_loss[bias] += loss.item()

                        # Gradient accumulation technique, which can achieve similar effects to using a large 'batch_size.
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer[bias].step()
                            scheduler[bias].step()
                            optimizer[bias].zero_grad()
                            global_step += 1 if bias == 0 else 0

                            if ((global_step + 1) % args.eval_step == 0) and (local_rank == 0 or local_rank == -1):

                                mean_loss = tr_loss[bias] * args.gradient_accumulation_steps / nb_tr_steps
                                mean_att_loss = tr_att_loss[bias] * args.gradient_accumulation_steps / nb_tr_steps
                                mean_rep_loss = tr_rep_loss[bias] * args.gradient_accumulation_steps / nb_tr_steps
                                # In this stage, there are no labels. The loss is obtained from the difference between 'T' and 'S' behavior_functions.
                                # The loss itself represents the similarity between the two models, so only the loss needs to be output.

                                result = {}
                                result['global_step'] = global_step
                                result['loss'] = mean_loss
                                result['att_loss'] = mean_att_loss
                                result['rep_loss'] = mean_rep_loss
                                output_eval_file = os.path.join(args.output_dir, "log.txt")
                                with open(output_eval_file, "a") as writer:
                                    logger.info("***** Eval results *****")
                                    for key in sorted(result.keys()):
                                        logger.info("  %s = %s", key, str(result[key]))
                                        writer.write("%s = %s\n" % (key, str(result[key])))

                                # Save a trained model
                                model_name = "step_{}_{}_{}".format(global_step, bias, WEIGHTS_NAME)
                                logging.info("** ** * Saving fine-tuned model ** ** * ")
                                # Only save the model it-self
                                model_to_save = student_model.student[bias]
                                output_model_file = os.path.join(args.output_dir, model_name)
                                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                self.tokenizer.save_vocabulary(args.output_dir)

                    nb_tr_steps += 1
                    pbar.update(1)

                if local_rank == 0 or local_rank == -1:
                    for bias in range(args.student_num):
                        model_name = "step_{}_{}_{}".format(global_step, bias, WEIGHTS_NAME)
                        logging.info("** ** * Saving fine-tuned model ** ** * ")
                        # Only save the model it-self
                        model_to_save = student_model.student[bias]
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        self.tokenizer.save_vocabulary(args.output_dir)
        if local_rank == 0 or local_rank == -1:
            for bias in range(args.student_num):
                logging.info("** ** * Saving final model ** ** * ")
                model_to_save = student_model.student[bias]

                final_dir_name = 'model_' + str(bias)
                final_dir_path = os.path.join(args.output_dir, final_dir_name)
                os.makedirs(final_dir_path)
                output_config_file = os.path.join(final_dir_path, CONFIG_NAME)
                output_final_file = os.path.join(final_dir_path, WEIGHTS_NAME)

                torch.save(model_to_save.state_dict(), output_final_file)
                model_to_save.config.to_json_file(output_config_file)
                self.tokenizer.save_vocabulary(final_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_num",
                        default=2,
                        type=int,
                        required=True)
    parser.add_argument("--distill_mode",
                        default=0,
                        type=int,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-4,
                        type=float, metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    '''parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")'''
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="Total number of devices")
    parser.add_argument("--node_num",
                        type=int,
                        default=0,
                        help="Number of current device")
    parser.add_argument("--ngpus_per_node",
                        type=int,
                        default=0,
                        help="Number of GPUs per node")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')
    parser.add_argument('--att_zero_like',
                        action='store_true',
                        help='Whether to zero some att when calculate att loss')

    # Additional arguments
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)

    args = parser.parse_args()
    args.world_size = args.world_size * args.ngpus_per_node
    logger.info('args:{}'.format(args))
    trainer = Trainer()
    if args.world_size > 1 and not args.no_cuda:
        torch.multiprocessing.spawn(trainer.train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        trainer.train(-1, args)
