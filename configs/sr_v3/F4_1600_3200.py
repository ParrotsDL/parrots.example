import os, sys
from yacs.config import CfgNode as CN

_C = CN()

############# Modify the following parameters for your local config ###############

# Color filter array pattern
_C.CFA_PATTERN = 'GRBG'
# Black levels (R, G, B)
_C.BLACK_LEVELS = [64, 64, 64]
# Saturation (or white level)
_C.SATURATION = 1023
# Resolution
_C.HEIGHT = 3024
_C.WIDTH = 4032
# Max/min ISO gains, can be set to exceed the sensor's physical limit
_C.ANALOG_GAIN_MIN = 16.0
_C.ANALOG_GAIN_MAX = 32.0
_C.DIGITAL_GAIN_MIN = 1.0
_C.DIGITAL_GAIN_MAX = 1.1
_C.READ_STD = 0.1 # empirical setting, in log scale

''' Meta noise model '''
def NOISE_MODEL(analog_gain, digital_gain, ch):
    S = {'R': 0.130654, 'G': 0.136505, 'B': 0.134011}
    R0 = {'R': 0.102993, 'G': 0.097464, 'B': 0.099197}
    R1 = {'R': 0.801315, 'G': 0.431676, 'B': 0.630061}
    shot_noise =  S[ch] * analog_gain * digital_gain
    read_noise = R0[ch] * ((analog_gain * digital_gain)**2) + R1[ch]
    return shot_noise, read_noise

'''
Finishing pipeline. Refer to the documentation for detailed usage.
'''
_C.FN = CN()

# Consensus transform mask
_C.FN.CT = CN()
_C.FN.CT.RADIUS = 3
_C.FN.CT.BLUR_RADIUS = 3
_C.FN.CT.BLUR_SIGMA = 2
_C.FN.CT.SCALE = 1.2
_C.FN.CT.BIAS = 0.0
_C.FN.CT.THRESH = 0.002
_C.FN.CT.TRAIN_THRESH = 0.02

# Chroma denoising
_C.FN.CD = CN()
_C.FN.CD.ON = True
_C.FN.CD.DR_RADIUS = 2
_C.FN.CD.DN_RADIUS = 3
_C.FN.CD.SIG_F = 50.0
_C.FN.CD.SIG_Y = 50.0
_C.FN.CD.SIG_C = 50.0
_C.FN.CD.DILATION = 4

# DoG sharpening
_C.FN.SP = CN()
_C.FN.SP.ON = True
_C.FN.SP.STRENGTH = 20.0
_C.FN.SP.RADIUS = 1
_C.FN.SP.SIGMA = 0.6
_C.FN.SP.ALPHA = 4.0

############## DO NOT modify unless you know what it means ##############

_C.SA = CN()
# Gaussian params
_C.SA.RADIUS = 2
_C.SA.SIGMA = 1.0
# Level 0 (LK refinement)
_C.SA.L0 = CN()
_C.SA.L0.TILE_SIZE = 64
_C.SA.L0.PADDING = 64
_C.SA.L0.MAX_ITER = 3
# Level 1
_C.SA.L1 = CN()
_C.SA.L1.DS_FACTOR = 2
_C.SA.L1.TILE_SIZE = 64
_C.SA.L1.PADDING = 64
_C.SA.L1.SEARCH_RADIUS = 1
# Level 2
_C.SA.L2 = CN()
_C.SA.L2.DS_FACTOR = 2
_C.SA.L2.TILE_SIZE = 64
_C.SA.L2.PADDING = 64
_C.SA.L2.SEARCH_RADIUS = 4
# Level 3
_C.SA.L3 = CN()
_C.SA.L3.DS_FACTOR = 4
_C.SA.L3.TILE_SIZE = 64
_C.SA.L3.PADDING = 64
_C.SA.L3.SEARCH_RADIUS = 4
# Level 4
_C.SA.L4 = CN()
_C.SA.L4.DS_FACTOR = 4
_C.SA.L4.TILE_SIZE = 32
_C.SA.L4.PADDING = 32
_C.SA.L4.SEARCH_RADIUS = 4

'''
Robust merging.
'''
_C.RM = CN()
_C.RM.TILE_SIZE = _C.SA.L0.TILE_SIZE
_C.RM.PADDING = _C.SA.L0.PADDING

