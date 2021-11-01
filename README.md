# metricssl_ecg

## Environment Setting

```
python == 3.9.7
pytorch == 1.10.0
```

## Self-Supervised Model Trainnig
```
```

## Baseline Model Training

To run ResNet18 model for the binary classification of PCWP, you can run:

```
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp_bin_resnet18
```

To run ResNet18 model for binary classification of PCWP with threshold 15 (originally threshold is 18), you can run:
```
python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_1 --pcwp-th 15 --epoch 500 --seed 1
```

To train and test ResNet18 model for the regression of PCWP (with normalization), you can run:

```
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode binary_class --model cnn --label pcwp --normalize-label --name pcwp_bin_resnet18
```

To test the performance of a model for PCWP binary classification, you can run:
```
CUDA_VISIBLE_DEVICES=1 python test.py --train-mode binary_class --model cnn --label pcwp --name pcwp_bin_resnet18
CUDA_VISIBLE_DEVICES=2 python test.py --name pcwp15_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp
```

For supervised metric learning, you can run:
```
CUDA_VISIBLE_DEVICES=1 python test.py --train-mode binary_class --model cnn --label pcwp --name pcwp_bin_resnet18 (need to be modified)
```
