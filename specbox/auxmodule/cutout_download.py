#!/usr/bin/env python
from pathlib import Path
from typing import Iterable
import time

import numpy as np
from PIL import Image
from astroquery.hips2fits import hips2fits
from astropy import units as u
from astropy.coordinates import Angle, Latitude, Longitude


EUCLID_CUTOUT_SURVEYS = (
    ("CDS/P/Euclid/Q1/color", "Euclid composite"),
)

NO_DATA_ERROR_MARKERS = (
    "Expecting value",
    "JSONDecodeError",
    "line 1 column 1",
    "char 0",
)


def get_cache_filename(buffer_dir, objid, survey_name):
    if buffer_dir is None or objid is None:
        return None
    base_dir = Path(buffer_dir)
    safe_survey = survey_name.replace("/", "_").replace(" ", "_")
    return base_dir / f"{objid}_{safe_survey}.jpg"


def save_cutout_to_cache(buffer_dir, objid, survey_name, image_data):
    cache_file = get_cache_filename(buffer_dir, objid, survey_name)
    if cache_file is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if len(image_data.shape) == 3:
        image = Image.fromarray(image_data, mode="RGB")
    else:
        image = Image.fromarray(image_data, mode="L")
    image.save(cache_file, "JPEG")


def load_cutout_from_cache(buffer_dir, objid, survey_name):
    cache_file = get_cache_filename(buffer_dir, objid, survey_name)
    if cache_file is None or not cache_file.exists():
        return None
    image = Image.open(cache_file)
    return np.array(image)


def is_no_data_error(exc):
    message = str(exc)
    return any(marker in message for marker in NO_DATA_ERROR_MARKERS)


def fetch_cutout(
    ra,
    dec,
    survey_name,
    size_arcsec=10,
    width=150,
    height=150,
    max_retries=3,
    retry_delays=(0.5, 1.0, 2.0),
):
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            return hips2fits.query(
                hips=survey_name,
                width=width,
                height=height,
                ra=Longitude(float(ra) * u.deg),
                dec=Latitude(float(dec) * u.deg),
                fov=Angle(float(size_arcsec) * u.arcsec),
                projection="CAR",
                format="jpg",
            )
        except Exception as exc:
            if attempt >= attempts - 1:
                if is_no_data_error(exc):
                    return None
                raise
            delay = retry_delays[min(attempt, len(retry_delays) - 1)]
            time.sleep(delay)
    return None


def is_valid_cutout_target(objid, ra, dec):
    if objid is None:
        return False
    if ra is None or dec is None:
        return False
    try:
        return not (np.isnan(ra) or np.isnan(dec))
    except TypeError:
        return False


def predownload_cutouts(
    records: Iterable[dict],
    buffer_dir,
    surveys=None,
    size_arcsec=10,
    progress_callback=None,
    request_delay_sec=0.15,
):
    selected_surveys = surveys or EUCLID_CUTOUT_SURVEYS
    records = list(records)
    total = len(records) * len(selected_surveys)
    done = 0
    downloaded = 0
    failed = 0
    skipped = 0
    no_data = 0
    Path(buffer_dir).mkdir(parents=True, exist_ok=True)

    for record in records:
        objid = record.get("objid")
        ra = record.get("ra")
        dec = record.get("dec")
        for survey_name, _ in selected_surveys:
            done += 1
            action = "skipped"
            try:
                if not is_valid_cutout_target(objid, ra, dec):
                    skipped += 1
                elif load_cutout_from_cache(buffer_dir, objid, survey_name) is not None:
                    skipped += 1
                else:
                    result = fetch_cutout(ra, dec, survey_name, size_arcsec=size_arcsec)
                    if result is not None:
                        save_cutout_to_cache(buffer_dir, objid, survey_name, result)
                        downloaded += 1
                        action = "downloaded"
                    else:
                        no_data += 1
                        action = "no_data"
                    if request_delay_sec and request_delay_sec > 0:
                        time.sleep(float(request_delay_sec))
            except Exception as exc:
                if is_no_data_error(exc):
                    no_data += 1
                    action = "no_data"
                else:
                    failed += 1
                    action = "failed"

            if progress_callback is not None:
                progress_callback(done, total, objid, action)

    return {
        "total": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "no_data": no_data,
        "failed": failed,
    }


def print_cli_progress(done, total, objid=None, action=None):
    if total <= 0:
        total = 1
    width = 32
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    tail = f" [{action}]" if action else ""
    objid_part = f" objid={objid}" if objid is not None else ""
    print(f"\rDownloading cutouts: |{bar}| {done}/{total}{tail}{objid_part}", end="", flush=True)
    if done >= total:
        print()
