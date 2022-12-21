#!/bin/bash

#SBATCH -n 1

#SBATCH --gres=gpu:1 
#SBATCH --constraint=maxwell
#SBATCH --job-name='inanim_0'
#SBATCH --time=03:00:00

#SBATCH -p nklab

#SBATCH --mail-type=END
#SBATCH --mail-user=kdobs@mit.edu

#SBATCH --output=output/output%j.out

# source /mindhive/nklab4/users/juliom/anaconda3/bin/activate
# conda activate torchenv-gpu

#source /mindhive/nklab4/users/kdobs/anaconda3/bin/activate

source /shared/venvs/py3.8-torch1.7.1/bin/activate
conda activate torchenv-gpu

CONFIG_FILE='face_vgg_large_modified.yaml'

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config_file $CONFIG_FILE --data_split "test" --batch_size 128 --iterator_seed 0 --maxout "True" --read_seed 0 --write_loss "False" --write_predictions "True"