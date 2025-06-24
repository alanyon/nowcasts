#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=40G
#SBATCH --ntasks=4
#SBATCH --output=/home/h04/alanyon/HAIC/nowcasts/test.out
#SBATCH --time=60
#SBATCH --error=/home/h04/alanyon/HAIC/nowcasts/test.err

CODE_DIR=global_nowcasts
export HTML_DIR=/home/users/andre.lanyon/public_html/HAIC/test
export SATDIR=/data/scratch/andre.lanyon/HAIC/sat_files
export DATADIR=/data/users/andre.lanyon/nowcasts
export MASSDIR=moose:/adhoc/projects/autosatarchive/adhoc/mtg_global_composites
export ENSDIR=/scratch/alanyon/HAIC/ens_files
export STEPS=6
export TIMESTEP=30
export INTERVAL=1
export NUM_VDTS=3
export START_TIME=20250623T0000
export END_TIME=20250623T0100

# Load scitools
module load scitools

# cd in code directory and run code
cd ${CODE_DIR}
python run_nowcast.py
