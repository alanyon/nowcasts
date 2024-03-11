#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=40G
#SBATCH --ntasks=4
#SBATCH --output=/home/h04/alanyon/nowcasts/test.out
#SBATCH --time=60
#SBATCH --error=/home/h04/alanyon/nowcasts/test.err

CODE_DIR=/net/home/h04/alanyon/nowcasts/avrdp2
export HTML_DIR=/net/home/h04/alanyon/public_html/HAIC/test_plots
export SATDIR=/scratch/alanyon/HAIC/sat_files
export DATADIR=/data/users/alanyon/GLOB_NOW
export STEPS=24
export INTERVAL=0.5
export CYLC_TASK_CYCLE_POINT=20230419T0000Z

# Load scitools
module load scitools

# cd in code directory and run code
cd ${CODE_DIR}
python run_nowcast.py
