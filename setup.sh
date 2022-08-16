#!/bin/bash

cd $HGCALML/modules
cd compiled
make -j 4
cd $HGCALML
git submodule update --init --recursive
