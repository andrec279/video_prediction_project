# Video Prediction Project

## Steps to access GCP Cluster Compute Node:

1. SSH into Greene: 
    1. `ssh greene` (must have ~/.ssh/config file configured properly to do this)
2. From Greene, SSH into Greene Burst login: `ssh burst`
3. From Greene Burst, two options:
    1. Request / log into shared burst node CPU to access files: ```srun --partition=interactive --account csci_ga_2572_2023sp_12 --pty /bin/bash```
    2. Request / log into GPU node for running jobs: ```srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_2023sp_12 --time=01:00:00 --pty /bin/bash```
4. Get our repo on your file system:
    1. Run `git clone git@github.com:andrec279/video_prediction_project.git`
    2. If permissions don’t exist, follow the steps in the following two links to configure SSH keys in your home directory
        1. Run `cd ~` to make sure you’re in your home directory
        2. Follow instructions to [generate an SSH private key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Make sure you’re following the Linux instructions and complete all steps EXCEPT the section “Generating a new SSH key for a hardware security key”.
        3. Follow instructions to [add your new SSH key to your Github profile](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account). Again, make sure to follow the instructions for Linux.
5. Run step 3.2 to launch a GPU instance for development / testing.
6. Once you are in bash-4.4, run `squeue -u $USER` and copy the the node ID under the “NODELIST(REASON)” column (e.g. b-3-106).
7. Open a new terminal window (on local computer) and run `vim ~/.ssh/config`.
8. Under the section “Host burstinstance burstinstancecontainer” update Hostname to the node ID.
9. In the Terminal window where you’re logged into your GPU instance, run `cd ~/video_prediction_project/demo`.
10. Run `sbatch demo.slurm`.
11. Go to VSCode and connect to burstinstancecontainer to open your IDE within the burst instance container.
