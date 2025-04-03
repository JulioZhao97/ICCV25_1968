# PubLayNet
# training
python train.py --data publaynet --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/publaynet --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data publaynet --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/publaynet --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data publaynet --model publaynet_best.pt --batch-size 64

# DocLayNet
# training
python train.py --data doclaynet --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/doclaynet --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data doclaynet --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/doclaynet --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data doclaynet --model doclaynet_best.pt --batch-size 64

# D4LA
# training
python train.py --data d4la --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/d4la --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data d4la --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/d4la --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data d4la --model d4la_best.pt --batch-size 64

# prima-lad
# training
python train.py --data prima-lad --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/prima-lad --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data prima-lad --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/prima-lad --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data prima-lad --model prima_best.pt --batch-size 64

# TableBank
# training
python train.py --data tablebank --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/tablebank --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data tablebank --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/tablebank --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data tablebank --model tablebank_best.pt --batch-size 64

# cord-v2 
# training
python train.py --data cord-v2 --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/cord-v2 --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data cord-v2 --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/cord-v2 --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data cord-v2 --model cordv2_best.pt --batch-size 64

# WTW
# training
python train.py --data wtw --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/WTW --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth1m.pt
python train.py --data wtw --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project public_dataset/WTW --plot 1 --optimizer SGD --lr0 0.02 --pretrain v10m-doclayout-docsynth300k.pt

# evaluation
python val.py --data wtw --model wtw_best.pt --batch-size 64