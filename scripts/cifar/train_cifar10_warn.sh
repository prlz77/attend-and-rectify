#!/usr/bin/env bash
# Should reach about 96.45% accuracy
python3 train_cifar_attention.py --depth 28 --width 4 --dataset CIFAR10 --batch_size 128 --lr 0.1 \
--epochs 200 --schedule 60 120 160 --lr_decay_ratio 0.2 --ngpu 1 --save ./logs/cifar10_attention \
--attention_depth 3 --attention_width 4 --attention_type softmax \
--reg_w 0.001