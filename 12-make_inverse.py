"""
====================
13. Inverse solution
====================

Compute and apply an inverse solution for each evoked data set.
"""

import os.path as op
import itertools
import logging

import mne
from mne.parallel import parallel_func
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mne_bids import make_bids_basename

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


@failsafe_run(on_error=on_error)
def run_inverse(subject, session=None):
    msg = 'Calculating inverse operator …'
    logger.info(gen_log_message(step=12, subject=subject, session=session,
                                message=msg))

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

    fname_ave = op.join(deriv_path, bids_basename + '-ave.fif')
    fname_fwd = op.join(deriv_path, bids_basename + '-fwd.fif')
    fname_cov = op.join(deriv_path, bids_basename + '-cov.fif')
    fname_inv = op.join(deriv_path, bids_basename + '-inv.fif')

    evokeds = mne.read_evokeds(fname_ave)
    cov = mne.read_cov(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    info = evokeds[0].info
    inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2,
                                             depth=0.8, rank='info')
    write_inverse_operator(fname_inv, inverse_operator)

    # Apply inverse
    msg = 'Calculating source time course for experimental conditions …'
    logger.info(gen_log_message(step=12, subject=subject, session=session,
                                message=msg))

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    method = config.inverse_method
    pick_ori = None

    inverse_str = f'inverse-{method}'
    hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.

    # Calculate inverse solutions for experimental conditions. Be sure not to
    # include any contrasts here (we will take care of those later).
    conditions = config.conditions
    for condition, evoked in zip(conditions, evokeds[:len(conditions)]):
        cond_str = 'cond-%s' % condition.replace(op.sep, '')
        fname_stc = op.join(deriv_path, '_'.join([bids_basename, cond_str,
                                                  inverse_str, hemi_str]))

        stc = apply_inverse(evoked=evoked,
                            inverse_operator=inverse_operator,
                            lambda2=lambda2, method=method, pick_ori=pick_ori)
        stc.save(fname_stc)

    # Now, calculate inverse solutions for the contrasts.
    contrasts = config.contrasts
    if contrasts:
        msg = 'Calculating source time course for contrasts …'
        logger.info(gen_log_message(step=12, subject=subject, session=session,
                                    message=msg))

    for contrast, evoked in zip(contrasts, evokeds[len(conditions):]):
        cond_1, cond_2 = contrast
        contrast_str = (f'contr-'
                        f'{cond_1.replace(op.sep, "")}-'
                        f'{cond_2.replace(op.sep, "")}')
        fname_stc = op.join(deriv_path, '_'.join([bids_basename, contrast_str,
                                                  inverse_str, hemi_str]))

        stc = apply_inverse(evoked=evoked,
                            inverse_operator=inverse_operator,
                            lambda2=lambda2, method=method, pick_ori=pick_ori)
        stc.save(fname_stc)


def main():
    """Run inv."""
    msg = 'Running Step 12: Compute and apply inverse solution'
    logger.info(gen_log_message(step=12, message=msg))

    parallel, run_func, _ = parallel_func(run_inverse, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.get_subjects(), config.get_sessions()))

    msg = 'Completed Step 12: Compute and apply inverse solution'
    logger.info(gen_log_message(step=12, message=msg))


if __name__ == '__main__':
    main()
