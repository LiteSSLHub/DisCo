
# DisCo
This is the repository of the paper ***[DisCo: Co-training Distilled Student Models for Semi-supervised Text Mining](https://arxiv.org/abs/2305.12074)***.

DisCo is a novel semi-supervised framework that addresses the challenges of both limited labeled data and computation resources in text mining tasks. 

We are able to produce light-weight model with DisCo that are 7.6&times; smaller and 4.8&times; faster in inference than the baseline PLMs while maintaining comparable performance.

DisCo-generated student models outperform the similar-sized models elaborately tuned in distinct tasks.

## Environment

- numpy
- torch
- transformers
- pyrouge
- rouge
- boto3

Run command below to install all the environment in need(**using python3**)

```shell
pip install -r requirements.txt
```

The pyrouge package requires additional installation procedures. If you need to run the extractive summarization task, please refer [this site](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu) to install pyrouge.

## Data

We provide all datasets used in our experiments:

- The dataset used for general distillation is **[WikiText-103](https://huggingface.co/datasets/wikitext)**.
- All the datasets used for downstream tasks are [**Agnews**, **Yahoo!Answer**, **DBpedia** and **CNN/DailyMail**](https://drive.google.com/drive/folders/1vnk81Xm4IgPSHZv71R-ygJmo-aiba_2P?usp=sharing). Please place the downloaded dataset folders into the ```./data``` folder.

## Usage

### Step 1:

Convert the downloaded WikiText-103 dataset into the format of BERT pre-training.

```shell
python src/pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \
                                          --bert_model ${BERT_BASE_DIR} \
                                          --do_lower_case \
                                          --epochs_to_generate 6 \
                                          --output_dir ${CORPUS_JSON_DIR}
```

-----

### Step 2:

Using the pre-training dataset obtained from step 1, we conduct knowledge distillation. We employ the BERT model as the teacher model. From this, we derive smaller student models that incorporate various layers of knowledge from the teacher. These student models serve as an effective initialization with model view for downstream tasks.

```shell
python src/general_distill.py --pregenerated_data ${CORPUS_JSON} \
                              --teacher_model ${BERT_BASE} \
                              --student_model ${STUDENT_CONFIG_DIR} \
                              --do_lower_case \
                              --train_batch_size 128 \
                              --gradient_accumulation_steps 1 \
                              --num_train_epochs 6 \
                              --ngpus_per_node 1 \
                              --output_dir ${GENERAL_TINYBERT_DIR} \
                              --student_num 6 \
                              --att_zero_like \
                              --distill_mode 1
```

For convenience, we provide all the general distilled student models used in Tables 2, 4, and 6.

- [6layer_2student_6epoch](https://drive.google.com/drive/folders/1kvcbSzKtB178AulCIIuqfAjGHQ9FslJp?usp=sharing) ($\rm S^{A6}$ and $\rm S^{B6}$)
- [4layer_2student_6epoch](https://drive.google.com/drive/folders/1hyvwi7wYcS-IFSQQN-3Utc1B77QCHqZi?usp=sharing) ($\rm S^{A4}$ and $\rm S^{B4}$)
- [2layer_2student_6epoch](https://drive.google.com/drive/folders/1awLz9yYZS1RRZ9rURkxf2Wwin9pg_Y54?usp=sharing) ($\rm S^{A2}$ and $\rm S^{B2}$)
- [6layer_4student_6epoch](https://drive.google.com/drive/folders/1vX3Jem-IoNDNLOGkxu-zjjTZ_UTDqE6x?usp=sharing) ($\rm S^{A2}$,  $\rm S^{B2}$, $\rm S^{C2}$ and $\rm S^{D2}$)

-----

### Step 3:

From the distilled student models generated in step 2, we apply the semi-supervised co-training method. This approach leverages unlabeled data and shares the diverse layer knowledge from the teacher model that are embedded in different student models. As a result, we obtain fine-tuned student models with performance comparable to that of the teacher model.

```shell
python src/train.py --dataset_name ${dataset_name} \
                    --train_batch_size 4 \
                    --test_batch_size 32 \
                    --val_batch_size 32 \
                    --num_workers 8 \
                    --tokenizer_name_or_path ${lm_name_or_path} \
                    --gradient_accumulation_steps 1 \
                    --encoder_name_or_path ${lm_name_or_path} \
                    --num_training_steps ${num_training_steps} \
                    --val_interval 100 \
                    --task_name ${task_name} \
                    --warmup_proportion 0.2 \
                    --root_dir ${root_dir} \
                    --ckpt_dir ${ckpt_dir} \
                    --lr ${lr} \
                    --student_num ${student_num} \
                    --lambdau 1 \
                    --rampup_rate 0.2 \
                    --cutoff_rate 0.2 \
                    --gpu 0 \
                    --do_consist \
                    --do_aug ${do_aug} \
                    --supervised_size ${supervised_size} \
                    --unsupervised_size 5000 \
                    --seed 33
```

All the example scripts can be found in `./script`

## Citation

```
@article{jiang2023disco,
  title={DisCo: Distilled Student Models Co-training for Semi-supervised Text Mining},
  author={Jiang, Weifeng and Mao, Qianren and Li, Jianxin and Lin, Chenghua and Yang, Weiyi and Deng, Ting and Wang, Zheng},
  journal={arXiv preprint arXiv:2305.12074},
  year={2023}
}
```