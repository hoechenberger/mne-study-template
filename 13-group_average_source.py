"""
=================================
14. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import os.path as op
import itertools
import logging

import pandas as pd

import mne
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def morph_stc(subject, session=None):
    msg = 'Morphing source estimates to fsaverage subject …'
    logger.info(gen_log_message(message=msg, subject=subject, session=session,
                                step=13))

    deriv_path = config.get_subject_deriv_path(subject=subject,
                                               session=session,
                                               kind=config.get_kind())

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.get_task(),
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space)

    method = config.inverse_method
    inverse_str = 'inverse-%s' % method
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph-fsaverage'

    morphed_stcs = list()

    conditions = config.conditions.copy()
    conditions.extend(config.contrasts)
    for condition in conditions:
        if condition in config.conditions:
            cond_str = 'cond-%s' % condition.replace(op.sep, '')
        else:  # This is a contrast of two conditions.
            contrast = condition
            cond_1, cond_2 = contrast
            cond_str = (f'contr-'
                        f'{cond_1.replace(op.sep, "")}-'
                        f'{cond_2.replace(op.sep, "")}')

        fname_stc = op.join(deriv_path, '_'.join([bids_basename, cond_str,
                                                  inverse_str, hemi_str]))
        fname_stc_fsaverage = op.join(deriv_path,
                                      '_'.join([bids_basename, cond_str,
                                                inverse_str, morph_str,
                                                hemi_str]))

        stc = mne.read_source_estimate(fname_stc)
        morph = mne.compute_source_morph(
            stc, subject_from=subject, subject_to='fsaverage',
            subjects_dir=config.get_fs_subjects_dir())
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(fname_stc_fsaverage)

        morphed_stc = pd.DataFrame(
            dict(subject=pd.Series(subject, dtype='string'),
                 session=pd.Series(session, dtype='string'),
                 condition=pd.Series(cond_str, dtype='string'),
                 stc=pd.Series(stc_fsaverage)))

        morphed_stcs.append(morphed_stc)
        del fname_stc, fname_stc_fsaverage, morphed_stc

    morphed_stcs = pd.concat(morphed_stcs, ignore_index=True)
    return morphed_stcs


@failsafe_run(on_error=on_error)
def main():
    """Run grp ave."""
    msg = 'Running Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))

    mne.datasets.fetch_fsaverage(subjects_dir=config.get_fs_subjects_dir())

    parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)

    morphed_stcs = parallel(run_func(subject, session)
                            for subject, session in
                            itertools.product(config.get_subjects(),
                                              config.get_sessions()))
    morphed_stcs = (pd.concat(morphed_stcs, ignore_index=True)
                    .sort_values(['subject', 'session', 'condition'])
                    .reset_index(drop=True))
    # We replace missing values with empty strings (could be anything, though),
    # as we want to iterate over the DataFrame via groupby() later – and
    # groupby() simply drops groups that have missing values in the grouping
    # variable(s), which we need to avoid.
    morphed_stcs = morphed_stcs.fillna('')

    method = config.inverse_method
    inverse_str = f'inverse-{method}'
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
    morph_str = 'morph-fsaverage'
    deriv_path = config.deriv_root

    conditions = config.conditions.copy()
    conditions.extend(config.contrasts)

    grouped = morphed_stcs.groupby(['session', 'condition'])
    for (session, condition), morphed_stc_subset in grouped:
        # If session is an empty string, we have inserted it to replace a
        # missing value. Make sure we set it to None again before creating the
        # basename, or we'll end up with a meaningless "ses-" entity.
        session = None if session == '' else session

        msg = f'Calculating average source estimates for: {condition}'
        logger.info(gen_log_message(message=msg, session=session, step=13))

        bids_basename = make_bids_basename(task=config.get_task(),
                                           session=session,
                                           acquisition=config.acq,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space)

        stc_avg = (morphed_stc_subset['stc'].sum() /
                   len(morphed_stc_subset['stc']))
        cond_str = 'cond-%s' % condition.replace(op.sep, '')

        fname_stc_avg = op.join(deriv_path, '_'.join(['average',
                                                      bids_basename, cond_str,
                                                      inverse_str, morph_str,
                                                      hemi_str]))
        stc_avg.save(fname_stc_avg)

    msg = 'Completed Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))


if __name__ == '__main__':
    main()
