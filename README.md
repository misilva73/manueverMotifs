## Finding maneuver motifs in vehicle telematics

This repository aims to support the work presented in paper *Finding maneuver motifs in vehicle telematics* by 
presenting all the analysis and figures present in the paper.

In this paper, we investigated a new way of identifying manoeuvres from vehicle telematics data, motif detection 
in time-series. We implemented a slightly modified version of the 
[*Extended Motif Discovery* (EMD) algorithm](https://github.com/misilva73/extendedMD), 
a classical variable-length motif detection algorithm for time-series and we applied it to the 
[UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/), 
a publicly available naturalistic driving dataset.

Particularly, we ran two different experiments. In the first, we aimed to identify acceleration and brakes from 
the longitudinal acceleration time-series and, in the second, we aimed to identify turns from the lateral acceleration 
time-series. The folders `experiments-lateral` and `experiments-longitudinal` contain the jupyter notebooks with the
results of these experiments.

Additionally, the folder `paper-figs` contains the plots used in the paper and the jupyter notebook that generated those
plots.