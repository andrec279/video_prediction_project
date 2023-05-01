# Instructions for Running Code on the Cluster
Working on Greene can be very confusing. One key concept to keep in mind is we refer to four different "workspaces" that are important to keep straight: local, Greene login, Greene compute, and Greene compute container. 

- Local is your computer.
- Greene login is the entry point into the Greene cluster that you access when you run `ssh greene` after properly configuring your .ssh/config file (see part 1 for what I mean by this). You should use this to request compute nodes for running your code as well as organizing your scratch folder, which is shared across all nodes on the Greene cluster.  
- Greene compute is the compute node you can request from Greene login. This is where you will launch your Singularity instance from.    
- Greene compute container is your Singularity instance that is running on Greene compute. This has your full conda environment with all your dependencies, and is what you will connect your VSCode instance to if you want to do remote development or run code remotely.

## Part 1: Configuring your .ssh/config file
To connect to the Greene cluster, you must first make sure you have your SSH settings configured properly on your local drive at `~/.ssh/config`. If this file does not exist on your computer yet, run `touch ~/.ssh/config`. Once you have verified this file exists, follow the steps outlined below:
1. Open your ~/.ssh/config file using your desired bash text editor (running `vim ~/.ssh/config` or `nano ~/.ssh/config` are both good options)
2. If your ~/.ssh/config file does not look like the one provided in this repository (see example_ssh_config_file), copy the contents of example_ssh_config_file into your ~/.ssh/config file but change all NetID references (where it says alc9635) to your own.
3. Save and close your ~/.ssh/config file.

## Part 2: One time setup of your project folder on Greene

1. SSH into Greene: `ssh greene` (this will work only if your ~/.ssh/config file is set up properly, see Part 1 if you haven't done this yet)
2. Go to your scratch folder: `cd /scratch/[YOUR_NETID_HERE]`
3. If you don't have this repository in your scratch folder, git clone it into your scratch folder.
4. If you already have the correct Dataset_Student/ folder on your Greene instance in your project folder, you can skip this step. Otherwise, download it from the Deep Learning Google Drive, then run the following in a **LOCAL** Terminal (replace with your netID): 

```
cd ~/Downloads
scp Dataset_Student_V2.zip [YOUR_NETID_HERE]@greene.hpc.nyu.edu:/scratch/[YOUR_NETID_HERE]/video_prediction_project
```

Then, go back to the Terminal that's connected to Greene login and run 

```
cd video_prediction_project
unzip Dataset_Student_V2.zip
```

you can delete the .zip file after this is done.

5. Similarly, make sure hidden folder on your Greene instance in your project folder. 

```
cd ~/Downloads
scp hidden_set_for_leaderboard_1.zip [YOUR_NETID_HERE]@greene.hpc.nyu.edu:/scratch/[YOUR_NETID_HERE]/video_prediction_project
```

Then, go back to the Terminal that's connected to Greene login and run 

```
cd video_prediction_project
unzip hidden_set_for_leaderboard_1.zip
```
you can delete the .zip file after this is done.


6. Run `cd NYU_HPC`, then from that folder run `make build`. This will tell bash to start building the overlays that will inject a conda environment with all our requirements into our Singularity instance. After you've done this once, you won't need to do it again (it takes a long time to run).

## Part 3: Launching a Greene compute node and starting Singularity instance on it
After setting up Parts 1 and 2, you can skip straight to Part 3 every time you want to connect to a Greene compute node to do remote development / run model training or any other GPU-intensive tasks.

1. SSH into Greene login: `ssh greene`
2. Go to the NYU_HPC folder within your project folder: `cd /scratch/[YOUR_NETID_HERE]/video_prediction_project/NYU_HPC`
3. Run `make getnode`. This will submit a job requesting a greene compute node.
4. Run `squeue -u $USER` to check the node ID of the greene compute node you were provisioned in the very last column titled "NODELIST (REASON)". It might take a while for an actual ID to show up here, so just re-run this command every minute or so until you see something like "gv014" in that column.
5. Copy the node ID, then open a different Terminal **LOCALLY**. On the local terminal, go to ~/.ssh/config file and paste the node ID under the section with the header "Host greenecompute greenecomputecontainer". For example, if your node ID is gv016, that section of your ~/.ssh/config file should look like this:
<img width="319" alt="image" src="https://user-images.githubusercontent.com/52364891/233734352-d0ffe797-06d2-4d49-ac8b-162625791b16.png">

6. Go back to your Terminal instance that's connected to Greene login, and run `exit` to close the Greene login connection. Then, run `ssh greenecompute` in that same terminal to connect to Greene compute.
7. Run `cd /scratch/[YOUR_NET_ID_HERE]/video_prediction_project/NYU_HPC`.
8. Run `make sing` to start your Singularity instance on your Greene compute node. You should see a message in your Terminal saying the Singularity instance was successfully launched.
9. From here, you have two options. See part 4 if you just want to run your code, and part 5 if you want to do full remote development on VSCode. 

## Part 4: Running your code in a Singularity instance
1. From a local Terminal window, run `ssh greenecomputecontainer`. This should connect you to your running Singularity instance at this point (your Terminal prompt should have "Singularity>" on the left-hand side), after which you can run `cd /scratch/[YOUR_NETID_HERE]/video_prediction_project` to cd into your project.
2. Initialize your conda environment: `conda activate /ext3/conda/dlproj` 
3. Run whatever .py scripts you like.

## Part 5: Remote Development in VSCode
**VERY IMPORTANT NOTE**: If doing remote development on the Greene compute node that you requested and set up in Part 3, note that the node will only stay activate for about 3-4 hours max before it gets shut down by HPC. So SAVE OFTEN to avoid losing your work accidentally.

1. In VSCode, go to the "Remote Explorer" tab. VSCode should already have populated this will all the locations you specified in your ~/.ssh/config file:
<img width="281" alt="image" src="https://user-images.githubusercontent.com/52364891/233735123-166d0528-c209-449c-a2bb-58972a1d96bc.png">
2. Hover over "greenecomputecontainer" and click the little plus-window looking icon on the right to launch a new VSCode window that is connected to your singularity instance running on your Greene Compute node:
<img width="350" alt="image" src="https://user-images.githubusercontent.com/52364891/233742794-4ffa1942-c282-46c1-a36a-186e60e7e723.png">

Note: This may require some additional configuration with VSCode to get working. For starters, make sure when you go to settings in VSCode and search "RemoteCommand" you have this option enabled:
<img width="1198" alt="image" src="https://user-images.githubusercontent.com/52364891/233743390-138ac452-ff9e-49e6-9021-547364e844fa.png">

3. Go to the "Extensions" tab in VSCode and download extensions for Python and Jupyter (should be the first options when you search either). 
4. In the top menu bar of VSCode, go to Terminal > New Terminal, then run `conda activate /ext3/conda/dlproj` to initialize your conda environment:
<img width="679" alt="image" src="https://user-images.githubusercontent.com/52364891/233743699-6f519b3d-080a-4534-afe8-de7da88f8f4f.png">

6. You can now click the "Open Folder" button on the left hand navigation of VSCode and enter /scratch/[YOUR_NETID_HERE]/video_prediction_project/ to get to your project folder, and you can now directly edit the code on your Singularity instance. Note that when editing Jupyter notebooks, you'll need to select the kernel that corresponds to your conda environment name, which should work fine after you've downloaded the Python and Jupyter extensions for VSCode in your remote instance.


## Part 6: Running our model

In your singularity instance do the following:

1. Make sure major packages are preinstalled
    - Run `pip install -r requirements.txt`

2. Use config.py file to set parameters for pretrain and finetune models. 
    - If pretraining for the first time or reconfiguring the parameters set `pretrain` as `True` in config.py file and change parameters accordingly. Remember than the output dimension of the pretrain model must match dim for input dimension of the finetune model.  
    - If using previously pretrained model, set `pretrain` as `False` and make sure `model_id` is the name of pretrained model you want to use.  
        - Note: previously pretrained model is saved in your current directory as `VICReg_pretrained_{time}.pth`

3. Pretrain and/or Finetune model 
    -  Run `python model_pipeline.py` in your singularity
    -  Once finished running new pretrained and finetuned models will be saved in your current directory. 
        -  Best finetuned model is selected from the epoch with best validation loss and saved in your current directory as `video_predictor_finetuned_best_val_{time}.pth`

4. Predict masks for hidden dataset 
    - Update the file names of those two models (pretrain and finetuned) in the `submission.ipynb`
        - run `ls -lah` to find the file names of the most recent models
    - Run all codes in `submission.ipynb`
    - Predicted masks for the hidden dataset will be a tensor of size (2000,160,240) and saved in `submitted_tensor_team12.pt`

    

