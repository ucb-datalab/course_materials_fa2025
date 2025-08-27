# AY 128/256 Course Materials - Fall 2025

This is the main GitHub repository for the course materials of AY 128 (Astronomy Data Science Lab) at UC Berkeley.

The Course website is located at [https://ucb-datalab.github.io/](https://ucb-datalab.github.io/).


## Setting up a work environment

### DataHub

For those enrolled in the course, you can head over to [astro.datahub.berkeley.edu](https://astro.datahub.berkeley.edu) which should have all the necessary dependencies installed for the course.

Then, open a Terminal window and clone this repo:

 ```
 git clone https://github.com/ucb-datalab/course_materials_fa2025.git
 ```

### Local Environment

You may wish to work directly from your laptop. To do so:

   1. Open up a terminal and then install [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

   2. Clone this repo: 
   
 ```
 git clone https://github.com/ucb-datalab/course_materials_fa2025.git
 ```

  3. Install the python packages

 ```
 cd course_materials_fa2025
 mamba env create -f environment.yml
 pip install --no-deps -r requirements.txt
 pip install -r requirements_deps.txt
 ```
 
  4. Activate the conda environment and install the kernel for Jupyter notebooks:
  
 ```
 conda activate astrods
 python -m ipykernel install --user --name=astrods
 ```
 
## Keeping it up to date
 
 We often make changes to the lecture material and so to sync with the latest, you should do a 
 
 ```
 git pull
 ```
 
Periodically in the `course_materials_fa2025` directory. If you've edited a file that is due to be changed, this can cause conflicts. Doing a `git stash` before the `pull` should help. But to avoid this, consider making a copy of any repo file you'd like to edit.

 
 
