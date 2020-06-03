# %%
import os

os.environ['MNE_BIDS_STUDY_CONFIG'] = "/home/parietal/rhochenb/Development/mne-study-template/tests/configs/config_ds000248.py"
os.environ['BIDS_ROOT'] = "/home/parietal/rhochenb/mne_data/ds000248"

%run /home/parietal/rhochenb/Development/mne-study-template/01-import_and_maxfilter.py

# %%
