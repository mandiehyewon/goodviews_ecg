CUDA_VISIBLE_DEVICES=0 python test.py --name demo_epoch100_1 --viewtype demos --load-step 20
CUDA_VISIBLE_DEVICES=0 python test.py --name clocslead_epoch100_dw1e-2 --viewtype clocslead --load-step 50
CUDA_VISIBLE_DEVICES=0 python test.py --name clocstime_epoch100_dw1e-2 --viewtype clocstime --load-step 50
CUDA_VISIBLE_DEVICES=0 python test.py --name attr_epoch100_dw1e-2 --viewtype attr --load-step 50
CUDA_VISIBLE_DEVICES=0 python test.py --name simclr_epoch7_batch16 --viewtype simclr --load-step 7