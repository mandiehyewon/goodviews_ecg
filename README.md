# Finding "Good Views" of Electrocardiogram Signals for inferring abnormalities in cardiac condition

## Environment Setting
Our code requires
```
python == 3.9.7
pytorch == 1.10.0
```

## Dataset
Chapman electrocardiogram dataset is available at [link](https://figshare.com/collections/ChapmanECG/4560497/2).
Download `ECGDataDenoised.zip` and unzip it. Place the unzipped `ECGDataDenoised` inside the `dir_csv` defined in `path_configs.yaml`.
Download `Diagnostics.xlsx` as well and place it inside the `dir_csv` defined in `path_configs.yaml` as well.


## Training and Evaluation
To train and evaluate the strategies in the paper, run this command:

```
CUDA_VISIBLE_DEVICES=0 python sup_train.py --epochs 100 --name base --embed-size 4 --viewtype sup --model cnn
```

Strategy 1 (CLOCS https://arxiv.org/pdf/2005.13249.pdf)
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name clocstime --viewtype clocstime
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --name clocslead --viewtype clocslead
```

Strategy 2 (SimCLR https://arxiv.org/pdf/2002.05709.pdf)

We prepared pre-augmented ECG signals. These pre-augmented version of ECG signal is  downlodable at [here](https://www.dropbox.com/s/k7s1xeibp2yg8xu/preaug.pickle?dl=0) in python `pickle`. Download `preaug.pickle` and place it inside the `dir_csv` path defined in `path_configs.yaml`.

```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 7 --name simclr --viewtype simclr --batch-size 16 --epochs 7
```

If you want to run without using pre-augmented ECG signals, you can run with `--no-preaug` argument as following command:

```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 7 --name simclr --viewtype simclr --batch-size 16 --epochs 7 --no-preaug
```

This will start from the beginning that creates pre-augmented ECG signals.

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

You can also refer to `runs` folder to use the above mentioned commands prepared in shell script.
