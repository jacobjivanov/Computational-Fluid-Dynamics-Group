#!/bin/bash
make comp
rm approx.txt
./"FFTW 1D Approximation" > "approx.txt"
python3 plot.py