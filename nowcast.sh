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
export STEPS=24
export TIMESTEP=30
export VDT_STR=20250623T2200

# Load scitools
module load scitools

# cd in code directory and run code
cd ${CODE_DIR}
# python extract_from_mass.py
# python run_nowcast.py
python delete_old_files.py