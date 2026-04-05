device_id=1
MODEL=mobilenetv2
ALPHA=123
PARTITION=homo
WD=1e-4
MU=0.001

HSCC_GAMMA=10
HSCC_BETA=5


CUDA_VISIBLE_DEVICES=$device_id python3 main.py \
    --dataset=cifar100 \
    --model=$MODEL \
    --approach=fedprox \
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
    --mu=$MU \
    --hscc \
    --hscc_gamma=$HSCC_GAMMA \
    --hscc_beta=$HSCC_BETA\
    --mu=$MU

