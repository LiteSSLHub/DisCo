export TOKENIZERS_PARALLELISM=false


gpu="0"

dataset_name="dbpedia"
task_name="dbpedia"
batch_size="--train_batch_size 4 --test_batch_size 32 --val_batch_size 32"

lm_name_or_path="experiments/6layer_2student_6epoch/"

lr="5e-3"
num_training_steps="50000"
gradient_accumulation_steps="1"
warmup_proportion="0.2"
student_num="2"
cutoff_rate="0.2"
rampup_rate="0.2"
lambdau="1"
do_aug="16"

root_dir="experiments/cnndm/consist/"
ckpt_dir="checkpoints1"
mkdir -p $root_dir

python src/train.py --dataset_name $dataset_name $batch_size --num_workers 8 --tokenizer_name_or_path $lm_name_or_path --gradient_accumulation_steps $gradient_accumulation_steps \
        --encoder_name_or_path $lm_name_or_path --num_training_steps $num_training_steps --val_interval 100 --task_name $task_name --warmup_proportion $warmup_proportion \
        --root_dir $root_dir --ckpt_dir $ckpt_dir --lr $lr --student_num $student_num --lambdau $lambdau --rampup_rate $rampup_rate --cutoff_rate $cutoff_rate \
        --gpu $gpu --do_consist --do_aug $do_aug --supervised_size 10 --unsupervised_size 5000 --seed 33