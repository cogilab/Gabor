### Codes for
### "Hard-wired visual filters for environment-agnostic object recognition" </br>

Min Jun Kang, Seungdae Baek, and Se-Bum Paik*

*Contact: sbpaik@kaist.ac.kr

### 1. System requirements
- MATLAB 2023a or later version
- Installation of the Deep Learning Toolbox (https://www.mathworks.com/products/deep-learning.html)
- Installation of the AlexNet (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)

### 2. Installation
- Download all files and folders or clone the repository
- Navigate to data/images/pacs and download the PACS dataset (https://www.kaggle.com/datasets/nickfratto/pacs-dataset)
- Expected Installation time is under 10 minutes
 
### 3. Instructions
1) Navigate to code_simul and run "main.m". 
- This code is a demo for training networks for a single domain pair ex. Photo to Sketch
- Set path0 as your porject path. ex. 'home/project_Gabor'
- For fast demo, you may change the "seed_list", which indicates the number of simulation networks, to a small number. ex. seed_list = 1:2;
- Many variables like paths can be modified in basicSettings.m in code_basic
- The simulation results will saved at results/results_simulation/

2) Navigate to code_figure and run proper figure codes
- Expected running time is about 5 minutes for F2, 1 hour of F3, 30 minutes for F4, but may vary by system conditions.

