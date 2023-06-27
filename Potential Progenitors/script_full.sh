#!/bin/bash

#SBATCH --job-name=toi-969-progenitors
#SBATCH --array=0-29 # Request N jobs with IDs equal to 0, ..., N
#SBATCH -t 0-23:00  # Request runtime of 23 hours DD-HH:MM
#SBATCH -C centos7  # Request only Centos7 nodes
#SBATCH -p sched_mit_mki # Run on sched_engaging_default partition sched_mit_mki
#SBATCH --mem-per-cpu=100  # Request 100MB of memory per CPU
#SBATCH -o output_%j.txt   # Redirect output to output_JOBID_TASKID.txt
#SBATCH -e error_%j.txt  # Redirect errors to error_JOBID_TASKID.txt
#SBATCH --mail-type=BEGIN,END  # Mail when job starts and ends
#SBATCH --mail-user=ofoo@mit.edu # Email recipient

## list of integers
declare -a arr=(1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30)

## load modules
module add anaconda3/2021.11

## install rebound
pip install --user rebound

## move to directory of this batch submit file
# cd /home/ofoo/v${arr[$SLURM_ARRAY_TASK_ID]}

## execute code 
python Potential_Progenitors.py $SLURM_ARRAY_TASK_ID
