#!/bin/bash
command="$@"
module load singularity
unset LD_LIBRARY_PATH
unset PYTHONPATH
cd 
# Chanaged by PZ to match new path
# singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/ceph/users/jkieseler/containers/deepjetcore3_latest.sif $command
# singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/ceph/users/jkieseler/containers/deepjetcore3_3.3.0.sif $command
# singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/home/pzehetner/containers/deepjetcore3_3.3.0.sif $command
# singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/home/pzehetner/containers/deepjetcore4_5ef28a3.sif $command
# singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/home/pzehetner/containers/deepjetcore4_06bc79b.sif $command
singularity  run  -B /mnt/ceph/users/ -B /mnt/home/ --nv /mnt/home/pzehetner/containers/deepjetcore_4_tag.sif $command
# this is this tag: deepjetcore4_abc9aee.sif
