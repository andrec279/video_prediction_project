# Video Prediction Project

## Steps to access GCP Cluster Compute Node:

1. SSH into Greene: 
    1. `ssh greene` (must have ~/.ssh/config file configured properly to do this)
2. From Greene, SSH into Greene Burst login: `ssh burst`
3. From Greene Burst, two options:
    1. Request / log into shared burst node CPU to access files: ```srun --partition=interactive --account csci_ga_2572_2023sp_12 --pty /bin/bash```
    2. Request / log into GPU node for running jobs: ```srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_2023sp_12 --time=01:00:00 --pty /bin/bash```
4. Run step 3.2 to launch a GPU instance for development / testing.
5. Make sure you're seeing bash-4.4 in your terminal indicating you're on the GPU instance.
6. If you haven't already, clone our repo to your GPU instance in the home directory:
    1. Make sure you are in home (~) directory. Run `cd ~` if unsure.
    2. Run `git clone git@github.com:andrec279/video_prediction_project.git`
    3. If permissions don’t exist, follow the steps in the following two links to configure SSH keys in your home directory
        1. Run `cd ~` to make sure you’re in your home directory
        2. Follow instructions to [generate an SSH private key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Make sure you’re following the Linux instructions and complete all steps EXCEPT the section “Generating a new SSH key for a hardware security key”.
        3. Follow instructions to [add your new SSH key to your Github profile](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account). Again, make sure to follow the instructions for Linux.
7. Go into the project directory: `cd ~/video_prediction_project`
8. Make sure your repo is up to date: `git pull`
9. Download the dataset into your folder:

```
pip install gdown
gdown https://drive.google.com/uc?id=1Ta34nFFoDqOKgoqoJ5fnE85i_9vcDfCr
```

11. Make sure you have your .py file you want to run in the root of your video_prediction_project folder.
12. Put /usr/bin/python3 [filename].py inside the quotes at the end of demo.slurm file, where [filename].py is the python file you want to run. You can add an additional line for any other python files you want to run as well.
13. Run `sbatch demo.slurm`.
14. Look at the logs corresponding to the job to see the output of your file.
