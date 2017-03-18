#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:01:36 2017

@author: raghu
"""

# Maintain static parameters required for project files

color_space = 'YCrCb'           # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, GRAY
orient = 9                      # 9 HOG orientations
pix_per_cell = 8                # 8 HOG pixels per cell
cell_per_block = 2              # HOG cells per block
hog_channel = 'ALL'             # Can be 0, 1, 2, or "ALL"
spatial_size = (32,32)          # 32 Spatial binning dimensions
hist_bins = 32	                # 32 Number of histogram bins
spatial_feat = True             # Spatial features on or off
hist_feat = True                # Histogram features on or off
hog_feat = True                 # HOG features on or off
overlap = 0.5                   # sliding window overlap percentage
heatmap_thresh = 4
