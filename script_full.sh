#!/bin/bash

#SBATCH --job-name=collision_tenmilyrs
#SBATCH --array=0-4 # Request N jobs with IDs equal to 0, ..., N
#SBATCH -t 0-12:00  # Request runtime of 12 hours DD-HH:MM
#SBATCH -C centos7  # Request only Centos7 nodes
#SBATCH -p sched_mit_hill # Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=500  # Request 500MB of memory per CPU
#SBATCH -o output_%j.txt   # Redirect output to output_JOBID_TASKID.txt
#SBATCH -e error_%j.txt  # Redirect errors to error_JOBID_TASKID.txt
#SBATCH --mail-type=BEGIN,END  # Mail when job starts and ends
#SBATCH --mail-user=ofoo@mit.edu # Email recipient

## list of integers
declare -a arr=(1  2  3  4  5)

## load modules
module add anaconda3/2021.11

## install rebound
pip install --user rebound

## move to directory of this batch submit file
cd /home/ofoo/v${arr[$SLURM_ARRAY_TASK_ID]}

## execute code 
python collision_testing_update.py $SLURM_ARRAY_TASK_ID

        
        

