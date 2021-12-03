# Finding "Good Views" of Electrocardiogram Signals for inferring abnormalities in cardiac condition

## Environment Setting
Our code requires
```
python == 3.9.7
pytorch == 1.10.0
```

## Dataset
Chapman electrocardiogram dataset is available in the following link: https://figshare.com/collections/ChapmanECG/4560497/2

## Training and Evaluation
To train and evaluate the strategies in the paper, run this command:

Strategy 1 (CLOCS https://arxiv.org/pdf/2005.13249.pdf)
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name clocstime --viewtype clocstime
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name clocslead --viewtype clocslead
```

Strategy 2 (SimCLR https://arxiv.org/pdf/2002.05709.pdf)
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 7 --name simclr --viewtype simclr --batch-size 16 --epochs 7
```

Strategy 3
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name demo --viewtype demos
```

Strategy 4
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name rhythm --viewtype rhythm
```

Strategy 5
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name attr --viewtype attr
```
You can also refer to runs folder.
