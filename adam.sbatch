#!/bin/bash
#SBATCH --job-name=adam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1

module purge
module load cuda/11.1.74
module load python/intel/3.8.6
module load anaconda3/2020.07

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ../../penv;
#pip install tensorboard
#pip install opencv-python
#export TORCH_CUDA_ARCH_LIST="compute capability"
#cd apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd ..
#python setup.py install --user
# pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#bash scripts/run.sh
EXPERIMENT_PATH="output/adam_worker4"
#mkdir -p $EXPERIMENT_PATH
DATASET_PATH=datasets/imagenet/ # was ~/Dataset/imagenet
DATASET=imagenet

python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_addr="127.0.0.1" --master_port=50000 flr/adam.py --data_path $DATASET_PATH --nmb_crops 2 6 --size_crops 32 16 --min_scale_crops 0.14 0.05 --max_scale_crops 1. 0.14 --use_pil_blur false --crops_for_assign 0 1 --temperature 0.1 --epsilon 0.05 --sinkhorn_iterations 3 --feat_dim 128 --nmb_prototypes 3000 --queue_length 0 --epoch_queue_starts 15 --epochs 2 --batch_size 32 --base_lr 4.8 --final_lr 0.0048 --freeze_prototypes_niters 313 --wd 0.000001  --warmup_epochs 0 --start_warmup 0.3 --dist_url "tcp://127.0.0.1:50000" --arch resnet50 --use_fp16 true --sync_bn apex --dataset $DATASET --dump_path $EXPERIMENT_PATH --tensorboard-log-dir tensorboard --syncbn_process_group_size 1

# syncbn_process_group_size is 8 by default
# base_lr was 4.8 and batch_size was 128 but too larg for NYU HPC
# size_crops was 224 96, but our image size is 48*48
