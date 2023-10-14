export TOKENIZERS_PARALLELISM=false

CORPUS_RAW="data/wikitext103/wiki.train.tokens"
BERT_BASE_DIR="bert-base-uncased"
CORPUS_JSON_DIR="data/wikitext103/trainingfile_6"


python src/pregenerate_training_data.py --train_corpus $CORPUS_RAW \
                  --bert_model $BERT_BASE_DIR \
                  --do_lower_case \
                  --epochs_to_generate 6 \
                  --output_dir $CORPUS_JSON_DIR