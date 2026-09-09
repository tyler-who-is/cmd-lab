# -*- coding: utf-8 -*-
"""
plate_solve.py — Astrometry.net plate solving + APASS catalog matching
for automatic standard star identification.
"""

import json
import time
import os
import tempfile
import warnings

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

ASTROMETRY_API = "https://nova.astrometry.net/api/"
API_KEY = "hvjbdvsgctupslac"
SOLVE_TIMEOUT = 300
POLL_INTERVAL = 5


def _session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def login(api_key=None):
    key = api_key or API_KEY
    s = _session()
    r = s.post(ASTROMETRY_API + "login",
               data={"request-json": json.dumps({"apikey": key})},
               timeout=30)
    r.raise_for_status()
    result = r.json()
    if result.get("status") != "success":
        raise RuntimeError(f"Astrometry.net 로그인 실패: {result}")
    return result["session"]


def upload(session_token, fits_path):
    s = _session()
    with open(fits_path, "rb") as f:
        r = s.post(
            ASTROMETRY_API + "upload",
            files={"file": (os.path.basename(fits_path), f,
                            "application/octet-stream")},
            data={"request-json": json.dumps({
                "session": session_token,
                "allow_commercial_use": "n",
                "allow_modifications": "n",
                "publicly_visible": "n",
                "scale_units": "degwidth",
                "scale_lower": 0.1,
                "scale_upper": 180,
                "scale_type": "ul",
            })},
            timeout=120,
        )
    r.raise_for_status()
    result = r.json()
    if result.get("status") != "success":
        raise RuntimeError(f"업로드 실패: {result}")
    return result["subid"]


def wait_for_job(subid, timeout=SOLVE_TIMEOUT, on_status=None):
    s = _session()
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = s.get(ASTROMETRY_API + f"submissions/{subid}", timeout=30)
            r.raise_for_status()
            data = r.json()
            jobs = data.get("jobs", [])
            if jobs and jobs[0] is not None:
                job_id = jobs[0]
                jr = s.get(ASTROMETRY_API + f"jobs/{job_id}", timeout=30)
                jr.raise_for_status()
                status = jr.json().get("status")
                if on_status:
                    on_status(f"job {job_id}: {status}")
                if status == "success":
                    return job_id
                if status == "failure":
                    raise RuntimeError(f"솔빙 실패 (job {job_id})")
        except requests.exceptions.SSLError:
            pass
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"솔빙 타임아웃 ({timeout}초)")


def get_wcs(job_id):
    s = _session()
    r = s.get(f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name
    try:
        with fits.open(tmp_path) as hdul:
            wcs_header = hdul[0].header
        return WCS(wcs_header)
    finally:
        os.unlink(tmp_path)


def solve_image(fits_path, api_key=None, on_status=None):
    if on_status:
        on_status("로그인 중...")
    token = login(api_key)
    if on_status:
        on_status("이미지 업로드 중...")
    subid = upload(token, fits_path)
    if on_status:
        on_status(f"솔빙 대기 중 (subid={subid})...")
    job_id = wait_for_job(subid, on_status=on_status)
    if on_status:
        on_status(f"WCS 다운로드 중 (job={job_id})...")
    wcs = get_wcs(job_id)
    return wcs


def pixels_to_radec(wcs, xy_list):
    xy = np.array(xy_list, dtype=float)
    radec = wcs.all_pix2world(xy, 0)
    return radec


def query_apass(ra_center, dec_center, radius_deg=0.5):
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = 5000
    coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)
    catalogs = Vizier.query_region(
        coord, radius=radius_deg * u.deg,
        catalog="II/336/apass9")
    if not catalogs:
        return None
    cat = catalogs[0]
    mask = np.isfinite(cat["Vmag"]) & np.isfinite(cat["Bmag"])
    cat = cat[mask]
    if len(cat) == 0:
        return None
    return cat


def match_stars(star_radec, catalog, max_sep_arcsec=10.0):
    star_coords = SkyCoord(ra=star_radec[:, 0] * u.deg,
                           dec=star_radec[:, 1] * u.deg)
    cat_coords = SkyCoord(ra=catalog["RAJ2000"], dec=catalog["DEJ2000"])
    idx, sep, _ = star_coords.match_to_catalog_sky(cat_coords)
    matches = []
    for i in range(len(star_coords)):
        if sep[i].arcsec <= max_sep_arcsec:
            j = idx[i]
            matches.append({
                "star_idx": i,
                "cat_idx": int(j),
                "sep_arcsec": float(sep[i].arcsec),
                "cat_V": float(catalog["Vmag"][j]),
                "cat_B": float(catalog["Bmag"][j]),
                "cat_ra": float(catalog["RAJ2000"][j]),
                "cat_dec": float(catalog["DEJ2000"][j]),
            })
    return matches


def auto_identify(fits_path, star_xy, api_key=None, on_status=None):
    wcs = solve_image(fits_path, api_key, on_status)
    if on_status:
        on_status("좌표 변환 중...")
    star_radec = pixels_to_radec(wcs, star_xy)
    ra_c = float(np.mean(star_radec[:, 0]))
    dec_c = float(np.mean(star_radec[:, 1]))
    ra_span = float(np.ptp(star_radec[:, 0]))
    dec_span = float(np.ptp(star_radec[:, 1]))
    radius = max(ra_span, dec_span) / 2.0 + 0.1
    radius = max(0.2, min(radius, 2.0))
    if on_status:
        on_status(f"APASS 카탈로그 조회 중 (r={radius:.1f}°)...")
    catalog = query_apass(ra_c, dec_c, radius)
    if catalog is None:
        raise RuntimeError("APASS 카탈로그에서 별을 찾지 못했습니다.")
    if on_status:
        on_status(f"카탈로그 {len(catalog)}개 별과 매칭 중...")
    matches = match_stars(star_radec, catalog)
    if on_status:
        on_status(f"매칭 완료: {len(matches)}개")
    return wcs, matches
