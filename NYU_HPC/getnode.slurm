#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/%u/video_prediction_project/NYU_HPC/logs/%x.out
#SBATCH --error=/scratch/%u/video_prediction_project/NYU_HPC/logs/%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80G 
#SBATCH -c 8

sleep 12h
