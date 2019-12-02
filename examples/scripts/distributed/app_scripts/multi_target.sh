numnodes=$1
rank=$2
masterip=$3

APPDIR=/home/qiangliu/Git/Mine/transformers/examples
pushd $APPDIR

#Kill Currently Running Jobs
sudo -H pkill python3.6

export PYTHONPATH=/home/qiangliu/Git/Mine/transformers/

# NCCL_SOCKET_IFNAME=^lo,docker0,veth NCCL_DEBUG=WARN \
NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \
python3.6 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=$numnodes \
    --node_rank=$rank \
    --master_addr=$masterip \
    --master_port=2222 \
    ${APPDIR}/run_glue.py \
    --input_train_dir /home/qiangliu/Git/Mine/transformers/examples/data/Caption/distill/train/l4_turing_quantus_malta_marco_sbs.tsv \
    --input_eval_dir /home/qiangliu/Git/Mine/transformers/examples/data/Caption/distill/eval \
    --model_type roberta \
    --model_name_or_path /home/qiangliu/Git/Mine/transformers/examples/model/multi_target/3_layers_distill \
    --task_name qp_multi_target \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=384 \
    --per_gpu_train_batch_size=384 \
    --learning_rate 2e-5 \
    --num_train_epochs 10.0 \
    --save_steps 2000 \
    --logging_steps 2000 \
    --fp16 \
    --output_dir /home/qiangliu/Git/Mine/transformers/examples/data/Caption/distill/roberta_distill/

