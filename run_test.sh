CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_2 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_3 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_4 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_5 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp18_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp --plot-prob

CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_1 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15 --plot-prob
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_2 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15 --plot-prob
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_3 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15 --plot-prob
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_4 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15 --plot-prob
CUDA_VISIBLE_DEVICES=3 python test.py --name pcwp15_bin_resnet18_epc500_5 --model cnn --last --train-mode binary_class --label pcwp --pcwp-th 15 --plot-prob