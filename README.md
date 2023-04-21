# Instructions for Running Code on the Cluster
Working on Greene can be very confusing. One key concept to keep in mind is we refer to four different "workspaces" that are important to keep straight: local (your computer), Greene login, Greene compute, and Greene compute container. 
    a. Local is self explanatory.
    b. Greene login is the entry point into the Greene cluster that you access when you run `ssh greene` after properly configuring your .ssh/config file (see part 1 for what I mean by this). You should use this to request compute nodes for running your code as well as organizing your scratch folder, which is shared across all nodes on the Greene cluster.
    c. Greene compute is the compute node you can request from Greene login. This is where you will launch your Singularity instance from.
    d. Greene compute container is your Singularity instance that is running on Greene compute. This has your full conda environment with all your dependencies, and is what you will connect your VSCode instance to if you want to do remote development or run code remotely.

## Part 1: Configuring your .ssh/config file
To connect to the Greene cluster, you must first make sure you have your SSH settings configured properly on your local drive at `~/.ssh/config`. If this file does not exist on your computer yet, run `touch ~/.ssh/config`. Once you have verified this file exists, follow the steps outlined below:
1. Open your ~/.ssh/config file using your desired bash text editor (running `vim ~/.ssh/config` or `nano ~/.ssh/config` are both good options)
2. If your ~/.ssh/config file does not look like the one provided in this repository (see example_ssh_config_file), copy the contents of example_ssh_config_file into your ~/.ssh/config file but change all NetID references (where it says alc9635) to your own.
3. Save and close your ~/.ssh/config file.

## Part 2: One time setup of your project folder on Greene

1. SSH into Greene: `ssh greene` (this will work only if your ~/.ssh/config file is set up properly, see Part 1 if you haven't done this yet)
2. Go to your scratch folder: `cd /scratch/[YOUR_NETID_HERE]`
3. If you don't have this repository in your scratch folder, git clone it into your scratch folder.
4. Run `cd video_prediction_project/NYU_HPC`, then from that folder run `make build`. This will tell bash to start building the overlays that will inject a conda environment with all our requirements into our Singularity instance. After you've done this once, you won't need to do it again (it takes a long time to run).

## Part 3: Launching a Greene compute node, starting Singularity instance on it, and connecting your VSCode instance
After setting up Parts 1 and 2, this is the only section you'll need to run each time you want to connect to a Greene compute node to do remote development / run model training or any other GPU-intensive tasks.

1. SSH into Greene login: `ssh greene`
2. Go to the NYU_HPC folder within your project folder: `cd /scratch/[YOUR_NETID_HERE]/video_prediction_project/NYU_HPC`
3. Run `make getnode`. This will submit a job requesting a greene compute node.
4. Run `squeue -u $USER` to check the node ID of the greene compute node you were provisioned in the very last column titled "NODELIST (REASON)". It might take a while for an actual ID to show up here, so just re-run this command every minute or so until you see something like "gv014" in that column.
5. Copy the node ID, then open a different Terminal **LOCALLY**. On the local terminal, go to ~/.ssh/config file and paste the node ID under the section with the header "Host greenecompute greenecomputecontainer". For example, if your node ID is gv016, that section of your ~/.ssh/config file should look like this:
<img width="319" alt="image" src="https://user-images.githubusercontent.com/52364891/233734352-d0ffe797-06d2-4d49-ac8b-162625791b16.png">
6. Go back to your Terminal instance that's connected to Greene login, and run `exit` to close the Greene login connection. Then, run `ssh greenecompute` in that same terminal to connect to Greene compute.
7. Run `cd /scratch/[YOUR_NET_ID_HERE]/video_prediction_project/NYU_HPC`.
8. Run `make sing` to start your Singularity instance on your Greene compute node.
9. In VSCode, go to the Remote Explorer tab. VSCode should already have populated this will all the locations you specified in your ~/.ssh/config file:
<img width="281" alt="image" src="https://user-images.githubusercontent.com/52364891/233735123-166d0528-c209-449c-a2bb-58972a1d96bc.png">

10. 

