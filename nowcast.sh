#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=40G
#SBATCH --ntasks=4
#SBATCH --output=/home/h04/alanyon/HAIC/nowcasts/test.out
#SBATCH --time=60
#SBATCH --error=/home/h04/alanyon/HAIC/nowcasts/test.err

CODE_DIR=global_nowcasts
export HTML_DIR=/net/home/h04/alanyon/public_html/HAIC/wcssp
export SATDIR=/scratch/alanyon/HAIC/sat_files
export DATADIR=/data/users/alanyon/GLOB_NOW
export ENSDIR=/scratch/alanyon/HAIC/ens_files
export STEPS=109
export TIMESTEP=30
export START_TIME=20220627T0000Z
export END_TIME=202206282300

# Load scitools
module load scitools

# cd in code directory and run code
cd ${CODE_DIR}
python run_nowcast_wcssp.py yes
