"""C6 RHS packing with two phase codes and explicit numerical checks.

Only one input per symmetry sector is allowed in a packet. C6 packets stay
within one (polarization, degree). With horizontal reflection, the sector
also includes (polarization+degree+m) mod 2 and both polarizations share a degree.
The two codes test numerical cross-talk; they do NOT independently measure
the residual of each unpacked surface solution.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from tmatrix_incident import regular_fields
from tmatrix_symmetry import c2x_plan, c2x_partner_column

VERSION = 3


def sector(mode, reflection=False):
    p,l,m=mode
    return m%6+(6*((p+l+m)%2) if reflection else 0)


def packet_id(indices):
    return hashlib.sha256(json.dumps(list(map(int, indices))).encode()).hexdigest()[:24]


def packets(modal, indices, reflection=False):
    """At most six C6 sectors or twelve C6/reflection sectors per degree."""
    groups = {}
    for index in dict.fromkeys(indices):
        p, l, m = modal[index]
        groups.setdefault((l,) if reflection else (p, l), []).append(index)
    result = []
    for key in sorted(groups):
        current, residues = [], set()
        for index in sorted(groups[key], key=lambda i: (modal[i][2],modal[i][0])):
            residue = sector(modal[index],reflection)
            if residue in residues:
                result.append(current)
                current, residues = [], set()
            current.append(index)
            residues.add(residue)
        if current:
            result.append(current)
    return result


def audit_packets(modal, order, reflection=False):
    """Both polarizations, low and intermediate degrees, independent references."""
    lookup = {mode: i for i, mode in enumerate(modal)}
    if reflection:
        return [[lookup[(p,l,m)] for m in range(min(l,5)+1) for p in (0,1)]
                for l in sorted({min(order,5),max(1,order//2)})]
    return [[lookup[(p, l, m)] for m in range(min(l, 5)+1)]
            for p in (0, 1) for l in sorted({min(order, 5), max(1, order//2)})]


def production_tasks(modal, indices, rejections=(), reflection=False):
    """Remember failures: C6+mirror -> C6 only -> individual RHS.

    A failed mirror packet does not invalidate rotational packing. Each
    smaller packet still has to pass both phase codes and leakage checks.
    """
    mirror_failed=set(); rotation_failed=set()
    for rejection in rejections:
        target=mirror_failed if rejection.get('reflection_sectors') else rotation_failed
        target.update(rejection['columns'])
    tasks=[]; singles=[]; groups={True:[],False:[]}
    for index in indices:
        if index in rotation_failed:
            singles.append(index)
        else:
            groups[bool(reflection and index not in mirror_failed)].append(index)
    for mirrored,group in groups.items():
        for packet in packets(modal,group,mirrored):
            if len(packet)==1:
                singles.extend(packet)
            else:
                tasks.append(('packed' if mirrored or not reflection else 'packed_rotation',packet))
    tasks.extend(('single',singles[i:i+2]) for i in range(0,len(singles),2))
    return tasks


def phase_codes(modal, indices, reflection=False):
    residues = [sector(modal[i],reflection) for i in indices]
    if not indices or len(residues) != len(set(residues)):
        raise ValueError('C6 packet must contain distinct m mod 6')
    return np.vstack((np.ones(len(indices), complex), np.exp(2j*np.pi*np.array(residues)/(12 if reflection else 6))))


def encode_fields(positions, modal, indices, ka, fields, field_function=regular_fields,
                  chunk_size=65536, reflection=False, field_cache=None):
    """Evaluate each mode once, reusing one temporary incident-field array.

    The two combined RHS arrays and one scratch field reside on the host;
    the worker still solves only two surface-current columns at a time.
    """
    codes = phase_codes(modal, indices,reflection)
    scales = []
    scratch = np.empty_like(fields[0])
    fields.fill(0)
    for j, index in enumerate(indices):
        if field_cache is not None:
            field_cache.fill(modal[index],scratch)
        else:
            for start in range(0, len(positions), chunk_size):
                scratch[start:start+chunk_size] = field_function(positions[start:start+chunk_size], modal[index], ka)
        scale = float(np.max(np.abs(scratch)))
        if not np.isfinite(scale) or scale < 1e-280:
            raise RuntimeError('regular-wave field underflow/nonfinite values in C6 packet')
        scales.append(scale)
        scratch /= scale
        for start in range(0, len(positions), chunk_size):
            values = scratch[start:start+chunk_size]
            fields[0, start:start+len(values)] += values
            fields[1, start:start+len(values)] += codes[1, j]*values
    combined = np.max(np.abs(fields), axis=(1, 2))
    if not np.isfinite(combined).all() or np.any(combined < 1e-280):
        raise RuntimeError('invalid combined C6 incident field')
    fields /= combined[:, None, None]
    return np.asarray(scales), combined


def decode_columns(projected, modal, indices, scales, combined, tolerance, reflection=False):
    """Unpack two projected far fields; reject inconsistent phase-code results."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('C6 tolerance must be finite and positive')
    codes = phase_codes(modal, indices,reflection)
    raw = np.asarray(projected)*np.asarray(combined)[:, None]
    residues = np.array([sector(mode,reflection) for mode in modal])
    columns, checks = {}, []
    for j, index in enumerate(indices):
        p, l, m = modal[index]
        mask = residues == sector(modal[index],reflection)
        a = np.where(mask, raw[0], 0)*(4*np.pi*1j**(l-p)*scales[j])
        b = np.where(mask, raw[1]/codes[1, j], 0)*(4*np.pi*1j**(l-p)*scales[j])
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-300)
        error = float(np.linalg.norm(a-b)/denominator)
        columns[index] = (a+b)*.5
        checks.append({'column': index, 'phase_code_relative_difference': error})
    occupied = np.isin(residues, [sector(modal[i],reflection) for i in indices])
    leakage = max(float(np.linalg.norm(row[~occupied])/max(np.linalg.norm(row), 1e-300)) for row in raw)
    finite = all(np.isfinite(column).all() for column in columns.values())
    passed = finite and np.isfinite(leakage) and leakage <= tolerance and all(
        np.isfinite(c['phase_code_relative_difference']) and c['phase_code_relative_difference'] <= tolerance
        for c in checks)
    return columns, {'version': VERSION, 'reflection_sectors':reflection,'columns': list(indices), 'checks': checks,
                     'empty_sector_relative_leakage': leakage, 'tolerance': tolerance,
                     'passed': bool(passed), 'individual_surface_residuals_measured': False}


def compare_references(columns, references, tolerance):
    checks = []
    for index, value in columns.items():
        reference = references[index]
        error = float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference), 1e-300))
        checks.append({'column': index, 'relative_column_error': error})
    return {'columns': checks, 'tolerance': tolerance,
            'passed': all(np.isfinite(c['relative_column_error']) and c['relative_column_error'] <= tolerance
                          for c in checks), 'sampled_not_an_operator_norm_bound': True}


def reflection_audit(modal, columns, tolerance):
    """Check sigma_h selection and sigma_v covariance against independent data."""
    permutation, signs, _, _=c2x_plan(max(l for p,l,m in modal))
    eta=np.array([(-1.)**(p+l+m+1) for p,l,m in modal])
    vertical_signs=signs*eta
    horizontal=[]; vertical=[]
    for i,value in sorted(columns.items()):
        norm=max(float(np.linalg.norm(value)),1e-300)
        horizontal.append({'column':i,'relative_forbidden_norm':float(np.linalg.norm(value[eta!=eta[i]])/norm)})
        target=int(permutation[i])
        if modal[i][2]>0 and target in columns:
            expected=c2x_partner_column(value,i,permutation,vertical_signs)
            error=float(np.linalg.norm(expected-columns[target])/max(np.linalg.norm(columns[target]),1e-300))
            vertical.append({'source':i,'target':target,'relative_column_error':error})
    maximum_h=max((x['relative_forbidden_norm'] for x in horizontal),default=float('inf'))
    maximum_v=max((x['relative_column_error'] for x in vertical),default=float('inf'))
    return {'horizontal_columns':horizontal,'vertical_column_pairs':vertical,
            'maximum_horizontal_error':maximum_h,'maximum_vertical_error':maximum_v,
            'tolerance':tolerance,'passed':bool(np.isfinite([maximum_h,maximum_v]).all() and
                max(maximum_h,maximum_v)<=tolerance),'sampled_not_an_operator_norm_bound':True}


def compatible_seed_parameters(source, target):
    """Packing changes the solve strategy, never the physical operator or basis."""
    def canonical(parameters):
        result = {k: v for k, v in parameters.items() if not k.startswith('c6_')}
        result['BEM_environment'] = {k: v for k, v in result.get('BEM_environment', {}).items()
            if k not in ('BEM_SO3_TMPDIR', 'BEM_SO3_BINARY_FP32', 'BEM_SO3_BINARY_FP64')}
        return result
    return canonical(source) == canonical(target)


def seed_columns(source, destination, parameters, directions, modal, tolerance):
    """Copy compatible individual/C2 checkpoints without modifying their source.

    Packed receipts are intentionally not imported across runs. They are
    resumed in their own directory with the exact packing fingerprint.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('--seed-from must be a different directory')
    manifest = json.loads((source/'manifest.json').read_text())
    old_directions = (source/'directions.bin').read_bytes()
    signature = hashlib.sha256(json.dumps(manifest['parameters'], sort_keys=True).encode()+old_directions).hexdigest()
    if signature != manifest['signature']:
        raise ValueError('seed manifest/directions checksum mismatch')
    if (not compatible_seed_parameters(manifest['parameters'], parameters)
            or old_directions != directions.tobytes()
            or manifest['modes'] != [list(mode) for mode in modal]):
        raise ValueError('seed columns belong to a different operator/order or tolerance')
    target = destination/'columns'
    target.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    permutation, signs, _, _ = c2x_plan(max(mode[1] for mode in modal))
    # Snapshot the stopped source; do not follow new files in an active build.
    for metadata_path in sorted((source/'columns').glob('*.json')):
        data_path = metadata_path.with_suffix('.npy')
        index = int(metadata_path.stem)
        if not data_path.exists() or not 0 <= index < len(modal):
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('derived_by') not in (None, 'C2x_half_turn') or 'c6_batch' in metadata:
            skipped += 1
            continue
        if metadata.get('derived_by') == 'C2x_half_turn':
            origin_path = source/'columns'/f'{metadata["source_column"]:06d}.json'
            if not origin_path.exists() or json.loads(origin_path.read_text()).get('derived_by') is not None:
                skipped += 1
                continue
        status = metadata['status']
        residual = max(status['strict_residual_x'], status['strict_residual_y'])
        value = np.load(data_path, allow_pickle=False)
        if (metadata['mode'] != list(modal[index]) or value.shape != (len(modal),)
                or not np.isfinite(value).all()
                or not np.isfinite([status['strict_residual_x'],status['strict_residual_y']]).all()
                or residual > tolerance):
            raise ValueError(f'invalid seed checkpoint {metadata_path.name}')
        if metadata.get('derived_by') == 'C2x_half_turn':
            origin = metadata['source_column']
            original = np.load(source/'columns'/f'{origin:06d}.npy', allow_pickle=False)
            if (not 0 <= origin < len(modal) or permutation[origin] != index
                    or not np.array_equal(value, c2x_partner_column(original, origin, permutation, signs))):
                raise ValueError(f'invalid C2 seed provenance {metadata_path.name}')
        output_data, output_meta = target/data_path.name, target/metadata_path.name
        if output_data.exists() and output_meta.exists():
            continue
        metadata['seed_source'] = {'directory': str(source), 'signature': signature}
        temporary = output_data.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, value)
        temporary.replace(output_data)
        temporary = output_meta.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary.replace(output_meta)
        copied += 1
    return {'source': str(source), 'source_signature': signature, 'copied_columns': copied,
            'skipped_packed_columns': skipped}
