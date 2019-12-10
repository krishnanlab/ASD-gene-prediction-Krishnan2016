#!/bin/bash
DIR="$( dirname "$(readlink -f "$0")" )"/..

echo "Downloading LibLINEAR..."
mkdir ${DIR}/bin
cd ${DIR}/bin
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/weights/liblinear-weights-2.30.zip
unzip liblinear-weights-2.30.zip && rm liblinear-weights-2.30.zip
cd liblinear-weights-2.30 
make

echo "Downloading brain-specific functional interaction network..."
mkdir -p ${DIR}/data/networks
cd ${DIR}/data/networks
wget http://giant.princeton.edu/static//networks/brain.gz

echo "Formatting network (this will take a while)..." 
gunzip brain.gz
python ${DIR}/src/format.py --network ${DIR}/data/networks/brain
