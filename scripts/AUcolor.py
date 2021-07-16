# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:34:52 2020

@author: ebbek
"""

def AUcolor():
    AU_Blue = [0,61,115]
    AU_Purple = [101,90,159]
    AU_Cyan = [55,160,203]
    AU_Green =  [139,173,63]
    AU_Yellow = [250,187,0]
    AU_Orange = [238,127,0]
    AU_Red = [226,0,26]
    AU_Magenta = [226,0,122]
    AU_Grey = [135,135,135]
    AU_Turkis = [0,171,164]
    AU_Dark_Blue = [0,37,70]
    AU_Dark_Purple = [40,28,65]
    AU_Dark_Cyan = [0,62,92]
    AU_Dark_Green = [66,88,33]
    AU_Dark_Yellow =  [99,75,3]
    AU_Dark_Orange = [95,52,8]
    AU_Dark_Red = [91,12,12]
    AU_Dark_Magenta = [95,0,48]
    AU_Dark_Grey = [75,75,74]
    AU_Dark_Turkis = [0,69,67]
    
    color_list = [AU_Blue,
                  AU_Purple,
                  AU_Green,
                  AU_Yellow,
                  AU_Orange,
                  AU_Cyan,
                  AU_Red,
                  AU_Magenta,
                  AU_Grey,
                  AU_Turkis,
                  AU_Dark_Blue,
                  AU_Dark_Purple,
                  AU_Dark_Cyan,
                  AU_Dark_Green,
                  AU_Dark_Yellow,
                  AU_Dark_Orange,
                  AU_Dark_Red,
                  AU_Dark_Magenta,
                  AU_Dark_Grey,
                  AU_Dark_Turkis]
    
    color_names = ['AU_Blue',
                  'AU_Purple',
                  'AU_Green',
                  'AU_Yellow',
                  'AU_Orange',
                  'AU_Cyan',
                  'AU_Red',
                  'AU_Magenta',
                  'AU_Grey',
                  'AU_Turkis',
                  'AU_Dark_Blue',
                  'AU_Dark_Purple',
                  'AU_Dark_Cyan',
                  'AU_Dark_Green',
                  'AU_Dark_Yellow',
                  'AU_Dark_Orange',
                  'AU_Dark_Red',
                  'AU_Dark_Magenta',
                  'AU_Dark_Grey',
                  'AU_Dark_Turkis']
    
    import numpy as np
    colors = np.array(color_list)/255
    return colors, color_names
