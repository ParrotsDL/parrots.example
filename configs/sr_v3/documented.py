import os, sys
from yacs.config import CfgNode as CN

_C = CN()

############# Modify the following parameters for your local config ###############

''' Global configurations '''
# Color filter array pattern of the bayer RAW. Though specified here, in theory
# a wrong pattern does not affect the result much.
_C.CFA_PATTERN = 'GRBG'
# Black levels (R, G, B)
_C.BLACK_LEVELS = [64, 64, 64]
# Saturation (or white level)
_C.SATURATION = 1023
# Camera resolution
_C.HEIGHT = 3024
_C.WIDTH = 4032
# Max/min ISO gains to sample
_C.ANALOG_GAIN_MIN = 0.0
_C.ANALOG_GAIN_MAX = 3.0
_C.DIGITAL_GAIN_MIN = 1.0
_C.DIGITAL_GAIN_MAX = 1.1
_C.READ_STD = 0.1 # empirical setting, in log scale

''' Meta noise model '''
# The noise model function receives the analog digital gains and returns the
# shot noise and read noise values, for different color channels.
def NOISE_MODEL(analog_gain, digital_gain, ch):
    S = {'R': 0.130654, 'G': 0.136505, 'B': 0.134011}
    R0 = {'R': 0.102993, 'G': 0.097464, 'B': 0.099197}
    R1 = {'R': 0.801315, 'G': 0.431676, 'B': 0.630061}
    shot_noise =  S[ch] * analog_gain * digital_gain
    read_noise = R0[ch] * ((analog_gain * digital_gain)**2) + R1[ch]
    return shot_noise, read_noise

'''
Finishing pipeline. Finishing only works at the testing stage.
'''
_C.FN = CN()

# Chroma denoising
_C.FN.CD = CN()
# Set to False to disable chroma denoising.
_C.FN.CD.ON = True
# Controlling the window size to analyse local structures. In general it does
# not need to be modified for 12M image.
_C.FN.CD.DR_RADIUS = 2
_C.FN.CD.DN_RADIUS = 3
# These sigma values, namely SIG_F, SIG_Y and SIG_C, controls the kernel spread
# for denoising. Set them to larger values for night scenes.
_C.FN.CD.SIG_F = 20.0
_C.FN.CD.SIG_Y = 20.0
_C.FN.CD.SIG_C = 20.0
# Larger dilation for denoising can effectively removes Chroma noise in the
# night. However, it may harms the overall image saturation and contrast.
_C.FN.CD.DILATION = 2

# DoG sharpening
_C.FN.SP = CN()
# Set to False to disable sharpening.
_C.FN.SP.ON = True
# Sharpening strength, defined as common.
_C.FN.SP.STRENGTH = 20.0
# Sharpening radius. Do not need to be changed in general
_C.FN.SP.RADIUS = 1
# Controling frequency band selection. These default values will choose the
# top frequencies to maximally increase image clarify. If it causes severe
# artifacts (e.g. large along-edge noise especially for high-ISO scenes), tune
# the sigma larger and alpha smaller (e.g. 0.4 and 6.0) to choose a lower
# frequency band. This trick can effectively removes high-frequency artifacts
# but may sacrifice image clarity a little bit.
_C.FN.SP.SIGMA = 0.1
_C.FN.SP.ALPHA = 8.0
# Consensus transform mask
_C.FN.CT = CN()
# The kernel radius for analysing local structures. In general it does not need
# to be modified for 12M image.
_C.FN.CT.RADIUS = 3
# Controling the blurring effect of consensus mask. Large blur introduces less
# sharpening at flat area. In general they do not need to be modified for 12M image.
_C.FN.CT.BLUR_RADIUS = 3
_C.FN.CT.BLUR_SIGMA = 2
# The scale and bias amplify the mask values for larger  sharpening strength.
# Do not need to modify in general.
_C.FN.CT.SCALE = 1.2
_C.FN.CT.BIAS = 0.0
# The threshold value is important. It balances sharpening effect at
# flat/detailed area. Larger threshold will introduce less sharpening on flat
# area (so as not to amplify noise), but may also miss the area with tiny
# details. On the other hand, small threshold effectively sharpens small
# details, but may exraggerate the noise in flat area. Empirically, we set this
# threshold larger (e.g. 0.005) for high ISO scenes.
_C.FN.CT.THRESH = 0.002

############## DO NOT modify unless you know what it means ##############

_C.SA = CN()
# Gaussian params
_C.SA.RADIUS = 2
_C.SA.SIGMA = 1.0
# Level 0 (LK refinement)
_C.SA.L0 = CN()
_C.SA.L0.TILE_SIZE = 32
_C.SA.L0.PADDING = 32
_C.SA.L0.MAX_ITER = 3
# Level 1
_C.SA.L1 = CN()
_C.SA.L1.DS_FACTOR = 2
_C.SA.L1.TILE_SIZE = 32
_C.SA.L1.PADDING = 32
_C.SA.L1.SEARCH_RADIUS = 1
# Level 2
_C.SA.L2 = CN()
_C.SA.L2.DS_FACTOR = 2
_C.SA.L2.TILE_SIZE = 32
_C.SA.L2.PADDING = 32
_C.SA.L2.SEARCH_RADIUS = 4
# Level 3
_C.SA.L3 = CN()
_C.SA.L3.DS_FACTOR = 4
_C.SA.L3.TILE_SIZE = 32
_C.SA.L3.PADDING = 32
_C.SA.L3.SEARCH_RADIUS = 4
# Level 4
_C.SA.L4 = CN()
_C.SA.L4.DS_FACTOR = 4
_C.SA.L4.TILE_SIZE = 16
_C.SA.L4.PADDING = 16
_C.SA.L4.SEARCH_RADIUS = 4

'''
Robust merging.
'''
_C.RM = CN()
_C.RM.TILE_SIZE = _C.SA.L0.TILE_SIZE
_C.RM.PADDING = _C.SA.L0.PADDING

