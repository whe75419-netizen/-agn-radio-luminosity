#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGN luminosity extraction: legacy calculation, luminosity-only output.

Install dependencies:
    python -m pip install numpy pandas astropy astroquery openpyxl
Run:
    python agn_luminosity.py

Output: name and rest-frame 6 GHz luminosity in units of 1e20 erg/s/Hz.
The spectral slope remains an internal quantity for the K-correction.

Scientific scope and limitations (preserved from the supplied calculation):
* NED 'Photometry Measurement' values are assumed to be Jy, without
  record-by-record unit verification or conversion.
* Near-6-GHz measurements are averaged; otherwise all selected radio
  records are fitted in log space, without weighting or epoch/resolution
  homogenization, explicit upper-limit filtering, or distinct-frequency checks.
* Slopes are rounded to two decimals before the K-correction. The near-6-GHz
  branch rejects abs(alpha) >= 4; the multi-frequency branch does not.
* An unavailable slope implicitly means alpha=0 for the luminosity calculation.
* Name deduplication is string-based only; aliases may refer to the same object.
These are documented legacy assumptions, not scientifically validated fixes.
Keep the actual input snapshots and adopted paper table: live queries cannot
guarantee reproduction of a historical database response.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astroquery.ipac.ned import Ned

# Optional network setting, e.g. AGN_NO_PROXY=ipac.caltech.edu
# Otherwise retain the user's existing proxy configuration.
if os.environ.get('AGN_NO_PROXY'):
    os.environ['no_proxy'] = os.environ['AGN_NO_PROXY']

# Warnings remain visible, including poorly conditioned fits.
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
LUMINOSITY_COLUMN = 'L_v_6GHz/10^20 erg·s^-1·Hz^-1'

def get_agn_name_list(limit=200):

    backup_list = [
        'NGC 1068', 'NGC 1275', 'M87', '3C 273', 'Cen A', 'Cygnus A',
        'Mrk 421', 'Mrk 501', 'OJ 287', 'BL Lacertae', 'NGC 4151',
        '3C 48', '3C 120', 'PKS 2155-304', 'AP Librae', '3C 279',
        '3C 454.3', 'TXS 0506+056', 'S5 0716+714', 'PKS 1510-089',

        'NGC 5548', 'NGC 4051', 'NGC 4388', 'NGC 3227', 'NGC 7469',
        'Mrk 335', 'Mrk 817', 'Mrk 509', 'NGC 1365', 'NGC 2110',
        'NGC 2992', 'NGC 3516', 'NGC 3783', 'NGC 4258', 'NGC 4593',
        'NGC 5506', 'NGC 7582', 'NGC 7172', 'NGC 1097', 'NGC 3081',

        'Mrk 3', 'Mrk 79', 'Mrk 279', 'Mrk 348', 'Mrk 478',
        'Mrk 841', 'Mrk 1040', 'Mrk 1095', 'NGC 5929', 'NGC 2273',

        'NGC 2639', 'NGC 3147', 'NGC 4579', 'NGC 4639', 'NGC 6814',

        'Ton 618', 'PG 1302-102', 'PG 0804+761', 'PG 1211+143',
        '3C 9', '3C 147', '3C 196', '3C 286', '3C 295', '3C 380',

        '1H 0323+342', 'Fairall 9', 'CTA 102', '3C 345', 'OX 169',

        'Fornax A', 'Pictor A', 'Hercules A', '3C 111', '3C 227',
        '3C 382', '3C 390.3', '3C 445',

        'Ark 120', 'Ark 564', 'IC 4329A', 'MCG-6-30-15',

        'PKS 0528+134', 'PKS 0537-441', 'PKS 0735+17',
        'PKS 0208-512', 'AO 0235+164',

        'OJ 248', 'OJ 049', 'ON 231', 'ON 325', 'OQ 530',

        '1ES 1959+650', '1ES 2344+514', 'H 1426+428',
        'PKS 2005-489',

        'NGC 1052', 'NGC 4261', 'NGC 3894', 'NGC 6500'
    ]

    extended_list = [
        'NGC 1386', 'NGC 1566', 'NGC 3393', 'NGC 4395',
        'NGC 4945', 'NGC 5128', 'NGC 5252', 'NGC 5643',
        'NGC 5728', 'NGC 6300', 'NGC 6810', 'NGC 7314',
        'NGC 7674', 'IC 5063', 'Circinus Galaxy',

        'Mrk 10', 'Mrk 78', 'Mrk 110', 'Mrk 205',
        'Mrk 231', 'Mrk 273', 'Mrk 290', 'Mrk 315',
        'Mrk 590', 'Mrk 766', 'Mrk 896', 'Mrk 926',
        'Mrk 1018',

        'PG 0052+251', 'PG 0844+349', 'PG 0953+414',
        'PG 1116+215', 'PG 1411+442', 'PG 1426+015',
        'PG 1613+658', 'PG 1700+518', 'PG 2130+099',
        'PG 2214+139',

        'I Zwicky 1', 'PHL 1092', 'PDS 456',

        '3C 47', '3C 109', '3C 207', '3C 212',
        '3C 245', '3C 287', '3C 334', '3C 336',
        '3C 432',

        '3C 15', '3C 17', '3C 31', '3C 33',
        '3C 61.1', '3C 79', '3C 83.1', '3C 133',
        '3C 219', '3C 236', '3C 264', '3C 293',
        '3C 303', '3C 305', '3C 310', '3C 321',
        '3C 338', '3C 353', '3C 388', '3C 401',
        '3C 433', '3C 438',

        '4C +29.30', '4C +12.50',

        'W Comae', 'S4 0954+65', 'S5 1803+784',
        'S5 2007+777', 'TXS 1508+056',

        '1ES 0229+200', '1ES 0347-121',
        '1ES 0414+009', '1ES 0806+524',
        '1ES 1011+496', '1ES 1218+304',
        '1ES 1312-423', '1ES 1741+196',

        'PKS 0048-09', 'PKS 0426-380',
        'PKS 0447-439', 'PKS 0558-504',
        'PKS 0903-57', 'PKS 1144-379',
        'PKS 1424-418', 'PKS 1502+106',
        'PKS 2142-75',

        'CTA 26', 'DA 193', 'NRAO 140', 'NRAO 530'
    ]

    full_list = list(dict.fromkeys(backup_list + extended_list))

    return full_list[:limit]

def fetch_agn_data_complete(source_names, target_freq_ghz=6.0):
    """Keep the legacy numerical calculation; return only name and luminosity."""
    if target_freq_ghz != 6.0:
        raise ValueError('This output schema is defined for 6 GHz only.')
    results = []
    for i, name in enumerate(source_names):
        print(f'[{i+1}/{len(source_names)}] 检索: {name} ...', end=' ')
        source_data = {'name': name, LUMINOSITY_COLUMN: np.nan}
        current_alpha = np.nan
        try:
            ned_obj = Ned.query_object(name)
            if len(ned_obj) == 0 or np.isnan(ned_obj['Redshift'][0]):
                print('未找到红移数据，跳过。')
                results.append(source_data)
                continue
            z = ned_obj['Redshift'][0]

            photo_table = Ned.get_table(name, table='photometry').to_pandas()
            photo_table['Pass'] = photo_table['Observed Passband'].astype(str)
            radio = photo_table[
                photo_table['Pass'].str.contains('GHz|MHz', case=False, na=False)
            ]
            if len(radio) > 0:
                radio = radio.dropna(subset=['Frequency', 'Photometry Measurement'])
                valid_radio = radio[
                    (radio['Frequency'] > 0) & (radio['Photometry Measurement'] > 0)
                ]
                if len(valid_radio) > 0:
                    freqs_ghz = valid_radio['Frequency'].values / 1e9
                    # Legacy assumption: raw measurements are Jy; NOT verified.
                    fluxes = valid_radio['Photometry Measurement'].values
                    tolerance = 0.01
                    exact_mask = np.abs(freqs_ghz - target_freq_ghz) <= tolerance
                    best_flux = np.nan

                    if np.any(exact_mask):
                        best_flux = np.mean(fluxes[exact_mask])
                        idx_low = np.abs(freqs_ghz - 1.4).argmin()
                        nu_l, f_l = freqs_ghz[idx_low], fluxes[idx_low]
                        if nu_l != target_freq_ghz and f_l > 0:
                            alpha = np.log10(best_flux / f_l) / np.log10(
                                target_freq_ghz / nu_l
                            )
                            current_alpha = round(alpha, 2) if abs(alpha) < 4 else np.nan
                    else:
                        if len(freqs_ghz) >= 2:
                            log_nu = np.log10(freqs_ghz)
                            log_S = np.log10(fluxes)
                            p = np.polyfit(log_nu, log_S, 1)
                            alpha = p[0]
                            log_S_6ghz = alpha * np.log10(target_freq_ghz) + p[1]
                            best_flux = 10**log_S_6ghz
                            current_alpha = round(alpha, 2)
                        else:
                            print('(射电点不足以拟合)', end=' ')

                    if not np.isnan(best_flux) and z > 0:
                        dl = cosmo.luminosity_distance(z).to('cm').value
                        if not np.isnan(current_alpha):
                            source_data[LUMINOSITY_COLUMN] = (
                                (4 * np.pi * dl**2 * best_flux * 1e-23)
                                / ((1 + z)**(1 + current_alpha)) / 1e20
                            )
                        else:
                            # Legacy fallback: implicitly alpha=0, not a measured slope.
                            source_data[LUMINOSITY_COLUMN] = (
                                (4 * np.pi * dl**2 * best_flux * 1e-23)
                                / (1 + z) / 1e20
                            )
            if np.isfinite(source_data[LUMINOSITY_COLUMN]):
                print('获得光度估计。')
            else:
                print('未获得有限光度值。')
        except Exception as e:
            print(f'失败! 原因: {e}')
        results.append(source_data)
        time.sleep(1)
    return pd.DataFrame(results, columns=['name', LUMINOSITY_COLUMN])


if __name__ == '__main__':
    print('说明：保留原算法及其未核查的 Jy 假设；此脚本不是科学验证后的修正版。')
    agn_list = get_agn_name_list(limit=200)
    if agn_list:
        df = fetch_agn_data_complete(agn_list)
        valid_count = int(np.isfinite(df[LUMINOSITY_COLUMN]).sum())
        print(f'\nAGN 光度结果：目标名称数 {len(df)}，有限光度值数 {valid_count}')
        print(df.head(20).to_string(index=False))
        output_file = 'AGN_200_Luminosity.xlsx'
        df.to_excel(output_file, index=False)
        print(f'数据已存入 {output_file}')
