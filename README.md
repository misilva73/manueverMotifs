# manueverMotifs - Finding maneuver motifs in vehicle telematics

***

## Overview

This repository aims to support the work presented in paper *Finding maneuver motifs in vehicle telematics*. 


output_folder = os.path.abspath(os.path.join(cwd, os.pardir, 'data-motifs'))
motif_file_name = 'motif_lat_acc_aggressive_trip.p'
dist_file_name = 'dist_lat_acc_aggressive_trip.npy'

motif_dic_list = pickle.load(open(os.path.join(output_folder, motif_file_name), 'rb'))
center_dist_mat = np.load(os.path.join(output_folder, dist_file_name))