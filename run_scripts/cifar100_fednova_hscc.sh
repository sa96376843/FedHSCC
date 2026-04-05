device_id=0
MODEL=mobilenetv2
ALPHA=00.5
PARTITION=noniid
WD=1e-4

HSCC_GAMMA=10
HSCC_BETA=5

CUDA_VISIBLE_DEVICES=$device_id main.py\
    --dataset=cifar100 \
    --model=$MODEL \
    --approach=fednova \
    --auto_aug \
    --lr=0.01 \
    --weight_decay=$WD \
    --epochs=10 \
    --n_comm_round=100 \
    --n_parties=10 \
    --partition=$PARTITION \
    --alpha=$ALPHA \
     --logdir='./logs/' \
    --datadir='./data/' \
    --ckptdir='./models/'\
    --print_interval=10 \
    --hscc \
    --hscc_gamma=$HSCC_GAMMA \
    --hscc_beta=$HSCC_BETA

