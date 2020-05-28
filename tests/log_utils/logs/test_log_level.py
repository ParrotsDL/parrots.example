import torch
import parrots
import os

fname = 'log_level.txt'

if os.path.exists(fname):
    os.remove(fname)

# ----------------------------------------
#
#  log to screen by default
#
# ----------------------------------------
# debug 1 will not display
parrots.log_utils.log_debug('debug 1')
parrots.log_utils.log_info('info 1')
parrots.log_utils.log_warn('warn 1')

parrots.log_utils.set_debug_log(True)

parrots.log_utils.log_debug('debug 2')
parrots.log_utils.log_info('info 2')
parrots.log_utils.log_warn('warn 2')

parrots.log_utils.set_debug_log(False)

# debug 3 will not display
parrots.log_utils.log_debug('debug 3')
parrots.log_utils.log_info('info 3')
parrots.log_utils.log_warn('warn 3')


# -------------------------
#
#  log to file
#
# -------------------------
parrots.log_utils.log_to_file(fname, False, False)

# debug 4 will not display
parrots.log_utils.log_debug('debug 4')
parrots.log_utils.log_info('info 4')
parrots.log_utils.log_warn('warn 4')

parrots.log_utils.set_debug_log(True)

parrots.log_utils.log_debug('debug 5')
parrots.log_utils.log_info('info 5')
parrots.log_utils.log_warn('warn 5')

parrots.log_utils.set_debug_log(False)

# debug 6 will not display
parrots.log_utils.log_debug('debug 6')
parrots.log_utils.log_info('info 6')
parrots.log_utils.log_warn('warn 6')


# ----------------------------------------
#
#  log to file and screen at the same time
#
# ----------------------------------------
parrots.log_utils.log_to_file(fname, False, True)

# debug 7 will not display
parrots.log_utils.log_debug('debug 7')
parrots.log_utils.log_info('info 7')
parrots.log_utils.log_warn('warn 7')

parrots.log_utils.set_debug_log(True)

parrots.log_utils.log_debug('debug 8')
parrots.log_utils.log_info('info 8')
parrots.log_utils.log_warn('warn 8')

parrots.log_utils.set_debug_log(False)

# debug 9 will not display
parrots.log_utils.log_debug('debug 9')
parrots.log_utils.log_info('info 9')
parrots.log_utils.log_warn('warn 9')
