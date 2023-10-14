export TOKENIZERS_PARALLELISM=false

CORPUS_JSON="data/wikitext103/trainingfile_6"
BERT_BASE="bert-base-uncased"
STUDENT_CONFIG_DIR="config/2layer_890m"
GENERAL_TINYBERT_DIR="experiments/2layer_4student_6epoch"


python src/general_distill.py --pregenerated_data $CORPUS_JSON \
                          --teacher_model $BERT_BASE \
                          --student_model $STUDENT_CONFIG_DIR \
                          --do_lower_case \
                          --train_batch_size 128 \
                          --gradient_accumulation_steps 1 \
                          --num_train_epochs 6 \
                          --ngpus_per_node 1 \
                          --output_dir $GENERAL_TINYBERT_DIR \
                          --student_num 6 \
                          --att_zero_like \
                          --distill_mode 1