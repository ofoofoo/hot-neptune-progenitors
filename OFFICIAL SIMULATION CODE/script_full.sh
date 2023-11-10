#!/bin/bash

#SBATCH --job-name=toi-969-progenitors
#SBATCH --array=0-149 # Request N jobs with IDs equal to 0, ..., N
#SBATCH -t 0-24:00  # Request runtime of 23 hours DD-HH:MM
#SBATCH -C centos7  # Request only Centos7 nodes
#SBATCH -p sched_mit_mki # Run on sched_engaging_default partition sched_mit_mki
#SBATCH --mem-per-cpu=100  # Request 100MB of memory per CPU
#SBATCH -o output_%j.txt   # Redirect output to output_JOBID_TASKID.txt
#SBATCH -e error_%j.txt  # Redirect errors to error_JOBID_TASKID.txt
#SBATCH --mail-type=BEGIN,END  # Mail when job starts and ends
#SBATCH --mail-user=ofoo@mit.edu # Email recipient

## list of integers
declare -a arr=(1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99  100  101  102  103  104  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  120  121  122  123  124  125  126  127  128  129  130  131  132  133  134  135  136  137  138  139  140  141  142  143  144  145  146  147  148  149  150)

## load modules
module add anaconda3/2021.11

## install rebound
pip install --user rebound

## move to directory of this batch submit file
# cd /home/ofoo/v${arr[$SLURM_ARRAY_TASK_ID]}

## execute code 
python Potential_Progenitors.py $SLURM_ARRAY_TASK_ID
