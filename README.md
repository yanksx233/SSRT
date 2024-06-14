# SSRT: Intra-and-Cross View Attention for Stereo Image Super-Resolution  

All codes will be released after paper is publised.

## Environment

```python
python >= 3.9.16
pytorch >= 1.12.0
cuda >= 11.4
natten
```

package `natten` can be installed with `pip install natten -f https://shi-labs.com/natten/wheels/cu113/torch1.12/index.html ` or refer to https://www.shi-labs.com/natten/.

## Quick Start

change work directory to `./code`

```
cd code
```

### Training

- x4

  ```

  python train.py --save_dir ssrt_4x --batch_size 32 --gpu_ids 3\
        --arch ssrt --num_feats 64 --num_heads 4 --depths 1*12 --kernel_size 9 --window_size 16 --num_cats 0 --upscale 4\
        --lr 2e-3 --wd 1e-4 --drop_path_rate 0.1 --beta2 0.9 --total_iter 200000\
        --scheduler_name cosine --periods 200e3 --min_lrs 1e-6 --warmup_steps 0 --mixup 0.7 --use_checkpoint

  ```

- x2

  ```

  python train.py --save_dir ssrt_2x --batch_size 32 --gpu_ids 4\
  --arch ssrt --num_feats 64 --num_heads 4 --depths 1*12 --kernel_size 9 --window_size 16 --num_cats 0 --upscale 2\
  --lr 2e-3 --wd 1e-4 --drop_path_rate 0.1 --beta2 0.9 --total_iter 200000\
  --scheduler_name cosine --periods 200e3 --min_lrs 1e-6 --warmup_steps 0 --mixup 0.7 --use_checkpoint

  ```

### Testing

- test on ETH3D and Middlebury

  - x4

  ```
  python test.py --save_dir ssrt_4x --upscale 4 --num_heads 4 --num_feats 64 --depths 1*12 --window_size 16 --arch ssrt --kernel_size 9 \
			--num_cats 4 --checkpoint ./SSRT_x4.tar --device cuda:0
  ```

  - x2

  ```
  python test.py --save_dir ssrt_2x --upscale 2 --num_heads 4 --num_feats 64 --depths 1*12 --window_size 16 --arch ssrt --kernel_size 9 \
			--num_cats 4 --checkpoint ./SSRT_x2.tar --device cuda:0
  ```

  
