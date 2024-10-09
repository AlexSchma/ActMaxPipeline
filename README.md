# ActMaxPipeline
Pipeline to 1st generate sequences via activation maximization.
Then 2nd use MEME to detect enriched motifs.
3rd use Tomtom to match enriched motifs to (A.thaliana) databases.
(4th visualize and summarize the results).

## 1st
Upload a model dict .pt file to the Data/model_weights directory.
Upload the model script file to Code/model_files.
In Code/ActMax/gen_seqs.py 

change the variables to import your model file and model dict.
if necessary change the get model function and other parameters (max_iter, early stopping, etc.)
if wished change the optimization objectives.
if wished change advanced parameters (additional entropy pennalty i.e.) at the bottom of the script

run the script

## 2nd & 3rd
MEME suite via docker desktop:
https://meme-suite.org/meme/doc/install.html?man_type=web#quick_docker

After that change the variable "directory name" in Code/MEME_Suite/bash_scripts/runMEME.sh & Code/MEME_Suite/bash_scripts/runtomtom.sh
to the name given by experiment_name in Code/ActMax/gen_seqs.py script

run the meme in docker
in the docker terminal, navigate to the directory of this repository
run sh-5.1$ bash Code/MEME_Suite/bash_scripts/runMEME.sh
after completion run
bash Code/MEME_Suite/bash_scripts/runTomTom.sh

## 4th coming
  
