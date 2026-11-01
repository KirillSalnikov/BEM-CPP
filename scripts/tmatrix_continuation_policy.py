"""Select a tested local continuation window; never fit off-centre T values."""
import math


def select_window(records, radii, models, tolerance=.01, calibration_fraction=.8):
    """Largest contiguous tested radius, then lowest required derivative order.

    Each model/radius must contain both signs and an explicit pole diagnostic.
    References used here are calibration controls. Fresh holdouts are required
    before accepting requested sizes or moving to a new expansion centre.
    """
    if (not 0 < tolerance < 1 or not 0 < calibration_fraction <= 1
            or not math.isfinite(tolerance + calibration_fraction) or not radii
            or any(not math.isfinite(r) or r <= 0 for r in radii)
            or list(radii) != sorted(set(radii)) or not models):
        raise ValueError('invalid continuation policy')
    names = [m['name'] for m in models]
    if len(set(names)) != len(names) or any(m['degree'] < 1 for m in models):
        raise ValueError('invalid model identities')
    rows = {}
    for r in records:
        key = r['model'], r['radius'], r['sign']
        if key in rows or key[0] not in names or key[1] not in radii or key[2] not in (-1, 1):
            raise ValueError('invalid or duplicate calibration point')
        error = r['acceptance_error']
        if error is not None and (not math.isfinite(error) or error < 0):
            raise ValueError('invalid continuation error')
        if error is None and r['pole_guard_passed']:
            raise ValueError('missing physical error for an eligible model')
        rows[key] = r
    if len(rows) != len(models) * len(radii) * 2:
        raise ValueError('incomplete bidirectional calibration')
    threshold = tolerance * calibration_fraction
    eligible = []
    diagnostics = []
    for model in models:
        selected, maximum = None, 0.
        for radius in radii:
            pair = [rows[model['name'], radius, sign] for sign in (-1, 1)]
            if any(not r['pole_guard_passed'] or r['acceptance_error'] is None
                   or r['acceptance_error'] > threshold for r in pair):
                break
            selected = radius
            maximum = max(maximum, *(r['acceptance_error'] for r in pair))
        item = dict(model=model['name'], degree=model['degree'], radius=selected,
                    maximum_calibration_error=maximum if selected is not None else None)
        diagnostics.append(item)
        if selected is not None:
            eligible.append(item)
    choice = min(eligible, key=lambda x: (-x['radius'], x['degree'], x['maximum_calibration_error'], x['model'])) if eligible else None
    return dict(status='needs_fresh_holdouts' if choice else 'need_smaller_radius_or_more_derivatives',
                selection=choice, alternatives=diagnostics, tolerance=tolerance,
                calibration_threshold=threshold, certified_continuous_interval=False,
                off_center_T_fitted=False, production_enabled=False)


def accept_holdouts(selection, records, center, target_stop):
    """A continuation centre may advance only after fresh interior controls.

    This returns a next-centre request, never substitutes a predicted T for
    the operator/moment solve at that new centre.
    """
    if selection['status'] != 'needs_fresh_holdouts' or not selection['selection']:
        raise ValueError('no calibrated continuation model')
    radius = selection['selection']['radius']
    expected = {sign * fraction * radius for sign in (-1, 1) for fraction in (.25, .75, .9)}
    if len(records) != len(expected) or {r['delta'] for r in records} != expected:
        raise ValueError('require all six fresh prescribed holdouts')
    if not math.isfinite(center + target_stop) or target_stop <= center:
        raise ValueError('positive centre advance required')
    for row in records:
        error = row['acceptance_error']
        if not row['independent_reference'] or (error is None and row['pole_guard_passed']):
            raise ValueError('invalid independent holdout receipt')
        if error is not None and (not math.isfinite(error) or error < 0):
            raise ValueError('invalid independent holdout receipt')
    passed = all(r['pole_guard_passed'] and r['acceptance_error'] is not None
                 and r['acceptance_error'] <= selection['tolerance'] for r in records)
    errors = [r['acceptance_error'] for r in records if r['acceptance_error'] is not None]
    return dict(status='new_center_required' if passed else 'shrink_or_increase_order', passed=passed,
                next_center=min(target_stop, center + radius) if passed else None,
                maximum_holdout_error=max(errors) if errors else None,
                checked_deltas=sorted(expected), certified_continuous_interval=False,
                limitation='Only sampled physical controls; mesh/L and untested sizes remain separate.')
