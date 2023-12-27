#!/bin/bash

# this script will fetch a random set of input files from remote. It's not advised that a lot of such files are copied from remote as they are large files. 

# so far this will only support fetching one such random data point from each of the 5 datasets of interest.

organList=("Dataset001_Anorectum" "Dataset002_Bladder" "Dataset003_CTVn" "Dataset004_CTVp" "Dataset005_Parametrium")

organList2=("Anorectum" "Bladder" "CTVn" "CTVp" "Parametrium")

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --raw"
    echo "  --testings"
    echo "  --all"
    exit 1
fi

function fetch_raw() {
    for organ in "${organList[@]}"
    do
        # assume each dataset has 100 samples
        randomSample=$((1 + $RANDOM % 100))
        randomSample=$(printf "%03d" "$randomSample")

        for type in "imagesTr" "labelsTr"
        do
            # make destination directory if it doesn't alread exist
            destination="tmp/nnUNet_raw/${organ}/${type}"
            mkdir -p $destination
            
            sample_name=$randomSample
            if [ $type = $images ]; then
                sample_name="${sample_name}_0000"
            fi

            source="shell1:/vol/biomedic3/bglocker/nnUNet/nnUNet_raw/${organ}/${type}/zzAMLART_${sample_name}.nii.gz"
            
            echo "[DEBUG]: copying files from"
            echo "         ${source}"
            echo "-------> ${destination}"

            scp $source $destination
        done
    done
}

function fetch_testings() {
    for organ in "${organList2[@]}"
    do
        # assume each dataset has 10 samples
        randomSample=$((1 + $RANDOM % 10))
        randomSample=$(printf "%03d" "$randomSample")

        # make destination directory if it doesn't alread exist
        destination="tmp/nnUNet_testing/${organ}"
        mkdir -p $destination

        source="shell1:/vol/biomedic3/bglocker/nnUNet/nnUNet_testing/predictions_${organ}/zzAMLIOV_${randomSample}.nii.gz"
        
        echo "[DEBUG]: copying files from"
        echo "         ${source}"
        echo "-------> ${destination}"

        scp $source $destination
    done    
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --raw)
            fetch_raw
            ;;
        --testings)
            fetch_testings
            ;;
        --all)
            fetch_raw
            fetch_testings
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done
