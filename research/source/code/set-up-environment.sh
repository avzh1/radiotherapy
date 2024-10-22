#!/bin/bash

if [ $# != 1 ]
then
    echo "Please provide the name of the directory from which you want to create a python venv for"
    exit -1
fi

DIR=$1

# obtain path to temporary directory to hold python's virtual environment
currPath=$(pwd)
venvPath=$currPath/.venv
# venvPath=$currPath/tmp/$DIR
# echo "DEBUG: creating venv path at ${venvPath}"
# mkdir -p $venvPath

# create virtual environment
export PENV=$venvPath
python3 -m virtualenv $PENV

# activate virtual environment
source $PENV/bin/activate
echo "verify the following are pointing to the correct location"
echo "       `which pip`"
echo "       `which python`"
echo "       `which python3`"

# install requirements from requirements.txt
pip install --upgrade pip
pip install -r $DIR/requirements.txt

echo "================"

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    source .venv/bin/activate
    # source tmp/$DIR/bin/activate
    cd $DIR
else
    echo "Enter the virtual environment with the following command:"
    echo "source .venv/bin/activate"
    # echo "source tmp/${DIR}bin/activate"
fi

echo "(exit with command 'deactivate')"
echo "================"