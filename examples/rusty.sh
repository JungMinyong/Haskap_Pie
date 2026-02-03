#!/bin/bash

## Recommended settings to run on a High Performance Cluster (HPC). This can also be used as an individual bash script.

## Following are slurm settings
##SBATCH -C genoa
#SBATCH -C icelake
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --job-name=haskap_pie2
#SBATCH --time=48:00:00   # hh:mm:ss for the job
##SBATCH -p preempt --qos=preempt

source ~/pie/bin/activate
module load intel-oneapi-compilers intel-oneapi-mpi

echo "Job started on $(date)"


## $1 is location of simulation snapshots: eg. /path/to/sim/box1/
## $2 is code type, one of ENZO, GADGET3, AREPO, GIZMO, ART, CHANGA, GEAR, RAMSES, manual
## $3 is same os save directory: eg. box1
## $4 is number of timesteps to skip, default should be 1.
SNAPDIR=/mnt/home/mjung/ceph/foggie_snaps
#SNAPDIR=/mnt/ceph/users/awright1/Maelstrom/2520_2
CODETYPE=ENZO
SAVEDIR=Maelstrom_resave
SKIP=1

python save_pfs.py $SNAPDIR $CODETYPE $SAVEDIR $SKIP
srun python save_particles_splited.py $SNAPDIR $CODETYPE $SAVEDIR $SKIP
python save_particles_merged.py $SNAPDIR $CODETYPE $SAVEDIR $SKIP
srun -n 16 python run_haskap.py $SNAPDIR $CODETYPE $SAVEDIR $SKIP


echo "Job finished on $(date)"