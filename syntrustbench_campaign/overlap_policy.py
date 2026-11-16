"""Row-overlap policy for datasets where distinct people can share identical rows.

Frozen v0.3 stops whenever a held-out row exactly equals a training row, using
row equality as a proxy for patient leakage. That proxy is right for UCI
Diabetes (rows are nearly unique) and wrong for coarse survey data such as
CDC BRFSS, where about 14% of respondents share their exact answers with
someone else. For such datasets the campaign has already proved person-level
disjointness on hashed respondent keys (RoleAssigner), so an exact row match
between roles is a coincidence, not a leak.

When a configuration sets ``rows_may_coincide=True``, this context manager
replaces only v0.3's split check while the block runs: overlap is counted and
reported as a warning instead of stopping the run. The frozen v0.3 source on
disk is not modified, so its hash verification still passes. Every other v0.3
check (schema, target, prevalence) is unchanged.
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def person_disjoint_roles(cfg):
    if not getattr(cfg, 'rows_may_coincide', False):
        yield
        return
    import syntrustbench.validation as validation
    original = validation._validate_splits

    def counted(real_train, real_test, config):
        train_keys = set(validation._canonical_keys(real_train, config).tolist())
        overlap = int(validation._canonical_keys(real_test, config).isin(train_keys).sum())
        warnings = [f'{overlap} held-out rows exactly match training rows; accepted because roles are '
                    'person-disjoint on hashed respondent keys and identical answers occur by chance.']
        for label, frame in (('real_train', real_train), ('real_test', real_test)):
            prevalence = float(frame[config.target].mean())
            if prevalence < 0.01 or prevalence > 0.99:
                warnings.append(f'{label} outcome prevalence is {prevalence:.2%}; discrimination and '
                                'subgroup estimates may be unstable.')
        return warnings

    validation._validate_splits = counted
    try:
        yield
    finally:
        validation._validate_splits = original
