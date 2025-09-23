#!/bin/sh
echo "Home: $HOME"

cd $1 && mm.py -l README.md
