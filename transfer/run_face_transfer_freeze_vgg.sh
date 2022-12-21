#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=12G   
#SBATCH --job-name='vgg_fr_face'
#SBATCH -t 8:00:00
#SBATCH -p nklab
#SBATCH --mail-type=END
#SBATCH --mail-user=mmpc2000@mit.edu
#SBATCH --output='./face/outputs/%A_%a.out'
#SBATCH --gres=gpu:QUADRORTX6000:1
# SBATCH --gres=gpu:GEFORCERTX2080TI:1
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=pascal|maxwell
#SBATCH --verbose


CONFIG_FILE='face_inanimate_400k_seed_modified.yaml'
BASEMODEL='face_inanimate_400k_seed'
NGPUS=1
# LEARNING_RATE=0.0001
USE_SCHEDULER=FALSE
RESTORE=TRUE
RESTORE_EPOCH=-1
READ_SEED=0
MAXOUT=TRUE

source /shared/venvs/py3.8-torch1.7.1/bin/activate
conda activate torchenv-gpu

CUDA_VISIBLE_DEVICES=0 python transfer_vgg_freeze.py --config_file $CONFIG_FILE --ngpus $NGPUS --read_seed $READ_SEED --maxout $MAXOUT --basemodel $BASEMODEL # --restore $RESTORE --restore_epoch $RESTORE_EPOCH # --learning_rate $LEARNING_RATE --use_scheduler $USE_SCHEDULER