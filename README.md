# ABD-Net adapted for Deep Learning Autonomous Vehicles

Forked repo from [ABD-Net](<https://github.com/TAMU-VITA/ABD-Net>).  
For details and original repo look [here](<https://github.com/theodorhusefest/ABD-Net/blob/master/README_ORIG.md>).

### Installation
 The first 2 steps are already done in the folder "scratch/husefest/".

1. Clone this repo
2. Create new virtualenv with ```python3 -m venv <env-name>```
3. Create a folder ```data``` containing the msmst17-dataset
4. Create a folder ```checkpoints``` for training
4. Activate virtualenv with ```source <env-name>/bin/activate```

### Training

1. Make sure you have activated virtualenv
2. Create new screen by running ```screen``` from terminal
3. Run ```bash train_on_msmt.sh```from terminal

### Evaluating 

1. Make sure you have activated virtualenv
2. Create new screen by running ```screen``` from terminal
3. Run ```bash evaluate_on_msmt.sh```from terminal


