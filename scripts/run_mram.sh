CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/MRAM_torch:$PYTHONPATH" python main.py \
    --lr 1e-3 \
    --batch_size 1024 \
    --encode_layer 2 \
    --decode_layer 2 \
    --l2 1e-4 \
    --ssm 0.01

