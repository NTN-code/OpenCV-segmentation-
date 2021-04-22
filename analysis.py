# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 23:06:41 2021

@author: NTN-code
"""

import matplotlib.pyplot as plt

x = [i for i in range(0,101)]

cluster = [0.49157, 0.4625631, 0.5033158, 0.4791671, 0.5031429, 0.4920233, 0.5452893, 0.5114428, 0.4926655, 0.5122271, 0.5659704, 0.4861123, 0.6366052, 0.3384234, 0.297146, 0.3675585, 0.2971842, 0.3019236, 0.2960754, 0.3492871, 0.3226282, 0.313967, 0.3389766, 0.3085514, 0.2902397, 0.3311466, 0.3279118, 0.2999228, 0.3198738, 0.2901785, 0.3723136, 0.3527576, 0.2971927, 0.3161892, 0.3450025, 0.3513893, 0.294971, 0.3228964, 0.3303595, 0.303053, 0.3226927, 0.306625, 0.304002, 0.2748644, 0.3091333, 0.2807031, 0.3044837, 0.3041573, 0.2985395, 0.4514953, 0.3548836, 0.3126162, 0.2920972, 0.3244271, 0.3063132, 0.3051555, 0.2965359, 0.3131122, 0.3344233, 0.3579169, 0.4615493, 0.317851, 0.2867944, 0.2933597, 0.3273759, 0.3267422, 0.3245205, 0.2950946, 0.3956134, 0.3105991, 0.3114095, 0.3344851, 0.3218024, 0.3783854, 0.3294574, 0.3034842, 0.2816669, 0.3211522, 0.3158008, 0.315932, 0.3471252, 0.3019894, 0.3173967, 0.3306102, 0.3218228, 0.2973733, 0.3120236, 0.3497876, 0.3412883, 0.3395207, 0.3450544, 0.3614786, 0.2846086, 0.3224063, 0.2976311, 0.3060653, 0.3003653, 0.3181174, 0.4270527, 0.3874292, 0.3109298]

watershed = [0.1060362, 0.1043299, 0.1049333, 0.1090904, 0.1061911, 0.1060613, 0.1065939, 0.1037722, 0.1065755, 0.1050124, 0.1060317, 0.1047719, 0.1090543, 0.1048297, 0.108002, 0.1064444, 0.1046143, 0.1088216, 0.1058909, 0.1064528, 0.1060472, 0.1055336, 0.105996, 0.1060165, 0.1075169, 0.1060541, 0.104406, 0.1042016, 0.1053782, 0.1386936, 0.1059443, 0.1059153, 0.1068505, 0.1087442, 0.1043057, 0.1055534, 0.1048054, 0.1056731, 0.1070905, 0.106655, 0.1077258, 0.1058994, 0.1083728, 0.1063887, 0.1051534, 0.1047998, 0.1068722, 0.1066416, 0.1069358, 0.1078242, 0.1071655, 0.105071, 0.1076794, 0.1046969, 0.1056233, 0.104717, 0.1054312, 0.1069754, 0.105843, 0.1099623, 0.1097302, 0.1101911, 0.1110469, 0.108626, 0.1084828, 0.106099, 0.1223348, 0.107798, 0.1079509, 0.1066698, 0.1056996, 0.109063, 0.1077196, 0.104661, 0.1071272, 0.1080921, 0.1085222, 0.1076085, 0.1066201, 0.1085893, 0.1120054, 0.1079236, 0.1057743, 0.1063242, 0.1108101, 0.1068696, 0.1089342, 0.1064516, 0.1074648, 0.1081965, 0.1115446, 0.111087, 0.1160006, 0.1066012, 0.1050421, 0.1072529, 0.1165659, 0.1085948, 0.1055377, 0.1044633, 0.1048192]

plt.plot(x, cluster)
plt.plot(x,watershed)
plt.bar(x,cluster, label="k-means")
plt.bar(x,watershed, label="watershed")
plt.legend()
plt.axis([0, 100, 0, 0.7])
plt.ylabel('Секунды')
plt.xlabel('Попытки')
plt.savefig('graph.png')
plt.show()


