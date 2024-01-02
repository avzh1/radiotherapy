#!/bin/bash

# obtain path to temporary directory to hold python's virtual environment
currPath=$(pwd)
venvPath=$currPath/../tmp/totseg-env
echo "DEBUG: creating venv path at ${venvPath}"
mkdir -p $venvPath

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
pip install -r requirements.txt

echo "================"
echo "Enter the virtual environment with the following command:"
echo "source ../tmp/totseg-env/bin/activate"
echo "(exit with command 'deactivate')"
echo "================"