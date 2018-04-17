# coding=utf-8

# to pre-define some constant variables

# the size of six output features
feature_size = [38, 19, 10, 5, 3, 1]

# 300 / feature_size
anchor_steps = [8, 16, 30, 60, 100, 300]

# the anchors' numbers of six branches,so there will be 11620 anchors together
anchors_num = [6, 6, 6, 6, 4, 4]

all_anchors_num = 11620

# the anchors' aspect ratios in six branches
anchors_ratio = [[1, 1, 2, 0.5, 3, 1./3],
                 [1, 1, 2, 0.5, 3, 1./3],
                 [1, 1, 2, 0.5, 3, 1./3],
                 [1, 1, 2, 0.5, 3, 1./3],
                 [1, 1, 2, 0.5],
                 [1, 1, 2, 0.5]]

# the anchors' scales, the default image's size is 300*300*3
# the first: ratio=1, sqrt(S_k*S_(k+1))
# the second: 0.07+(k-1)*(0.87-0.1)/(6-1), k=1...6
"""anchors_scales = [[0.13, 0.07],
                  [0.30, 0.23],
                  [0.46, 0.39],
                  [0.62, 0.55],
                  [0.79, 0.71],
                  [0.95, 0.87]]"""

anchors_size = [[39, 21],
                [90, 69],
                [138, 108],
                [186, 165],
                [237, 213],
                [285, 261]]





