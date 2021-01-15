#!/bin/bash
#$ -l h_rt=4:00:00  #time needed
#$ -pe smp 8 #number of cores
#$ -l rmem=20G #number of memery
#$ -o ScalableML/q2_output.txt #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M rgeorge5@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit  --driver-memory 30g --executor-memory 12g --master local[8] --conf  spark.driver.maxResultSize=5g   ScalableML/Q2_code.py