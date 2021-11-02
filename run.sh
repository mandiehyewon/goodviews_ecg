CUDA_VISIBLE_DEVICES=2 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp18_bin_resnet18_epc500_1 --epoch 500 --seed 1
CUDA_VISIBLE_DEVICES=3 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp18_bin_resnet18_epc500_2 --epoch 500 --seed 2
CUDA_VISIBLE_DEVICES=4 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp18_bin_resnet18_epc500_3 --epoch 500 --seed 3
CUDA_VISIBLE_DEVICES=5 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp18_bin_resnet18_epc500_4 --epoch 500 --seed 4
CUDA_VISIBLE_DEVICES=6 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp18_bin_resnet18_epc500_5 --epoch 500 --seed 5
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_1 --pcwp-th 15 --epoch 500 --seed 1
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_2 --pcwp-th 15 --epoch 500 --seed 2
CUDA_VISIBLE_DEVICES=2 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_3 --pcwp-th 15 --epoch 500 --seed 3
CUDA_VISIBLE_DEVICES=2 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_4 --pcwp-th 15 --epoch 500 --seed 4
CUDA_VISIBLE_DEVICES=3 python train.py --train-mode binary_class --model cnn --label pcwp --name pcwp15_bin_resnet18_epc500_5 --pcwp-th 15 --epoch 500 --seed 5

CUDA_VISIBLE_DEVICES=0 python train.py --train-mode regression --model cnn --label pcwp --pcwp-th 15 --epoch 500 --seed 1 --name pcwp15_reg_resnet18_epc500_1
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode regression --model cnn --label pcwp --pcwp-th 15 --epoch 500 --seed 2 --name pcwp15_reg_resnet18_epc500_2
CUDA_VISIBLE_DEVICES=2 python train.py --train-mode regression --model cnn --label pcwp --pcwp-th 15 --epoch 500 --seed 3 --name pcwp15_reg_resnet18_epc500_3
CUDA_VISIBLE_DEVICES=3 python train.py --train-mode regression --model cnn --label pcwp --pcwp-th 15 --epoch 500 --seed 4 --name pcwp15_reg_resnet18_epc500_4
CUDA_VISIBLE_DEVICES=4 python train.py --train-mode regression --model cnn --label pcwp --pcwp-th 15 --epoch 500 --seed 5 --name pcwp15_reg_resnet18_epc500_5

CUDA_VISIBLE_DEVICES=2 python train.py --train-mode regression --model cnn --label pcwp --epoch 500 --seed 1 --name pcwp18_reg_resnet18_epc500_1
CUDA_VISIBLE_DEVICES=3 python train.py --train-mode regression --model cnn --label pcwp --epoch 500 --seed 2 --name pcwp18_reg_resnet18_epc500_2
CUDA_VISIBLE_DEVICES=4 python train.py --train-mode regression --model cnn --label pcwp --epoch 500 --seed 3 --name pcwp18_reg_resnet18_epc500_3
CUDA_VISIBLE_DEVICES=0 python train.py --train-mode regression --model cnn --label pcwp --epoch 500 --seed 4 --name pcwp18_reg_resnet18_epc500_4
CUDA_VISIBLE_DEVICES=1 python train.py --train-mode regression --model cnn --label pcwp --epoch 500 --seed 5 --name pcwp18_reg_resnet18_epc500_5


CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_2 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_3 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_4 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_5 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15

CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_reg_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_reg_resnet18_epc500_2 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_reg_resnet18_epc500_3 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_reg_resnet18_epc500_4 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_reg_resnet18_epc500_5 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15

CUDA_VISIBLE_DEVICES=2 python test.py --train-mode regression --model cnn --label pcwp --last --seed 1 --name pcwp18_reg_resnet18_epc500_1
CUDA_VISIBLE_DEVICES=3 python test.py --train-mode regression --model cnn --label pcwp --last --seed 2 --name pcwp18_reg_resnet18_epc500_2
CUDA_VISIBLE_DEVICES=4 python test.py --train-mode regression --model cnn --label pcwp --last --seed 3 --name pcwp18_reg_resnet18_epc500_3
CUDA_VISIBLE_DEVICES=0 python test.py --train-mode regression --model cnn --label pcwp --last --seed 4 --name pcwp18_reg_resnet18_epc500_4
CUDA_VISIBLE_DEVICES=1 python test.py --train-mode regression --model cnn --label pcwp --last --seed 5 --name pcwp18_reg_resnet18_epc500_5