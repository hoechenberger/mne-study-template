study_name = 'ds000246'
runs = ['01']
subjects_list = ['0001']
ch_types = ['meg']
reject = {'mag': 4e-12}
conditions = ['standard', 'deviant', 'button']
contrasts = [('standard, deviant')]
decode = True
crop = (100, 200)
daysback = -365 * 110
