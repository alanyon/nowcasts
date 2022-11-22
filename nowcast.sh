#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=20G
#SBATCH --ntasks=4
#SBATCH --output=/home/h04/alanyon/HAIC/test.out
#SBATCH --time=30
#SBATCH --error=/home/h04/alanyon/HAIC/test.err

CODE_DIR=/net/home/h04/alanyon/first_guess_TAFs/python
export HTML_DIR=/net/home/h04/alanyon/public_html/HAIC/test_plots
export SATDIR=/scratch/alanyon/HAIC/sat_files
export DATADIR=/data/users/alanyon/GLOB_NOW
export STEPS=8

# Load scitools
module load scitools/production-os45-1

# cd in code directory and run code
cd ${CODE_DIR}
python run.py
