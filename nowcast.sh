#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --ntasks=4
#SBATCH --output=/home/users/andre.lanyon/nowcasts/nowcasts/test.out
#SBATCH --time=5
#SBATCH --error=/home/users/andre.lanyon/nowcasts/nowcasts/test.err

CODE_DIR=/home/users/andre.lanyon/nowcasts/nowcasts/global_nowcasts
export HTML_DIR=/home/users/andre.lanyon/public_html/HAIC/test
export SCRATCH_DIR=/data/scratch/andre.lanyon/HAIC
export MASSDIR=moose:/adhoc/projects/autosatarchive/adhoc/mtg_global_composites
export ENSDIR=/scratch/alanyon/HAIC/ens_files
export STEPS=4
export TIMESTEP=30
export VDT_STR=20250504T1200

# Load scitools
module load scitools

# cd in code directory and run code
cd ${CODE_DIR}
# python extract_from_mass.py
python run_nowcast.py
# python delete_old_files.py