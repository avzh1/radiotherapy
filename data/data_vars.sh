#!/bin/bash

# run with . data_vars.sh

PROJECT_DIR=$(git rev-parse --show-toplevel)
OLD_DIR='/vol/bitbucket/az620/radiotherapy/'

# get the directory of the script
DATA_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create an empty array to store the new environment variables
new_env_vars=()

export PROJECT_DIR=$PROJECT_DIR
export OLD_DIR=$OLD_DIR
export DATA_DIR=$DATA_DIR

new_env_vars+=(PROJECT_DIR)
new_env_vars+=(OLD_DIR)
new_env_vars+=(DATA_DIR)

# Get names of all directories present in DATA_DIR and export them
for dir in $(find $DATA_DIR -maxdepth 1 -type d ! -name "$(basename $DATA_DIR)"); do
    export $(basename $dir)=$dir
    new_env_vars+=($(basename $dir))
done

# assume we have an nnUNet_raw folder. Therefore, inside we have the different organ
# classes. Get names of all directories present in nnUNet_raw and export them
for dir in $(find $nnUNet_raw -maxdepth 1 -type d ! -name "$(basename $nnUNet_raw)"); do
    IFS="_" read -ra parts <<< "$(basename $dir)"
    # access the parts of the string using ${parts[index]}
    export ${parts[1]}="$(basename $dir)"
    # add new environment variable to list
    new_env_vars+=(${parts[1]})
done

# images and ground truth label directories
export data_trainingImages=imagesTr
export data_trainingLabels=labelsTr
new_env_vars+=(data_trainingImages data_trainingLabels)

# check if the script is called with an argument indicating we should print the
# environment variables
if [ -n "$1" ]; then
    # Iterate over the array and export the new environment variables
    for var in "${new_env_vars[@]}"; do
        echo "$var: ${!var}"
    done
fi