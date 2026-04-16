#!/bin/bash

# Default values
NAME=rdp_train
MODEL=rdp

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set GPU devices and NUM_GPUS
case $MODEL in
    "rdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        NUM_GPUS=4
        ;;
    "cma")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "cma_plus")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "seq2seq")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "seq2seq_plus")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "navdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    "flownav_static")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "flownav_dyn")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "flownav_mix")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    *)
        echo "Error: Unsupported model type: $MODEL"
        exit 1
        ;;
esac

# Check if NUM_GPUS is set
if [[ -z $NUM_GPUS ]]; then
    echo "Error: NUM_GPUS is not set"
    exit 1
fi


export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO

# navdp/flownav family use torchrun for consistent distributed launch behavior
if [[ "$MODEL" == "navdp" || "$MODEL" == "flownav_static" || "$MODEL" == "flownav_dyn" || "$MODEL" == "flownav_mix" ]]; then
    echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12345 \   # 注意：如果在多机环境下运行，需要修改master_addr和master_port以确保正确通信
        scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL"
else
    echo "Using python to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    python scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL"
fi
