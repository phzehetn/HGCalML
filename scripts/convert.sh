#!/bin/bash


list_subdirectories() {
    local parent_dir=$1
    find "$parent_dir"  -mindepth 1 -maxdepth 1 -type d
}


DIRECTORY="/mnt/ceph/users/pzehetner/Paper/Test/granular_pu_tests/pu_test_events"

# Check if HGCALML is set
if [ -z "$HGCALML" ]; then
    echo "Error: HGCALML environment variable is not set."
    exit 1
fi


if [ -d "$DIRECTORY" ]; then
    cd $DIRECTORY
    subdirs=$(list_subdirectories "$DIRECTORY")

    for subdir in $subdirs; do
	
	if [ -d "$subdir/preclustered" ]; then
	    echo $(basename "$subdir")
	    echo "Already preclustered"
	    continue
	elif [ -f $subdir/00009.djctd ]; then
	    cp $DIRECTORY/filelist.txt $subdir
	    convertDJCFromSource.py -i $subdir/filelist.txt -o $subdir/preclustered -c TrainData_PreSnowflakeNanoML --gpu -n 1
	elif [ -f $subdir/00000.djctd ]; then
	    cp $DIRECTORY/reduced_filelist.txt $subdir
	    convertDJCFromSource.py -i $subdir/reduced_filelist.txt -o $subdir/preclustered -c TrainData_PreSnowflakeNanoML --gpu -n 1
	fi
    done
fi

