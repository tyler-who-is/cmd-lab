#!/usr/bin/env python3
"""
make_demo_data.py -- generates a small synthetic B/V FITS dataset (with bias,
dark, flat, and light frames, plus a fake open-cluster star field with a
realistic-looking main sequence + turnoff) in the folder layout that
cmd_pipeline.py expects. Use this to rehearse the "one button" workflow
before the real observing night -- no telescope data needed.

Usage:
    python make_demo_data.py --out data
"""
import argparse
import os
import numpy as np
from astropy.io import fits

rng = np.random.default_rng(42)


def write_fits(path, data):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def make_star_field(n_stars=150, size=800):
    """Fake cluster: stars cluster toward the center (real members) plus a
    sprinkling of uniform 'field' stars near the edges. Flux in V and B is
    assigned along a fake main sequence + a handful of brighter turnoff/giant
    stars, so the resulting CMD has a recognizable bent shape."""
    cx, cy = size / 2, size / 2

    # cluster members: concentrated near center, main-sequence + turnoff color-mag relation
    n_mem = int(n_stars * 0.7)
    r = rng.rayleigh(scale=size * 0.12, size=n_mem)
    theta = rng.uniform(0, 2 * np.pi, n_mem)
    x_mem = cx + r * np.cos(theta)
    y_mem = cy + r * np.sin(theta)
    # main sequence: fainter stars are redder (higher B-V), turnoff stars are the brightest/bluest
    t = rng.uniform(0, 1, n_mem) ** 1.5  # bias toward fainter (lower mass) stars
    v_mag_true = 8 + 8 * t                 # V from 8 (bright turnoff) to 16 (faint dwarfs)
    bv_true = 0.0 + 1.3 * t + rng.normal(0, 0.03, n_mem)  # bluer at turnoff, redder for dwarfs
    b_mag_true = v_mag_true + bv_true

    # field stars: uniformly scattered, unrelated color-mag relation (contamination)
    n_field = n_stars - n_mem
    x_fld = rng.uniform(0, size, n_field)
    y_fld = rng.uniform(0, size, n_field)
    v_mag_fld = rng.uniform(9, 16, n_field)
    bv_fld = rng.uniform(-0.1, 1.6, n_field)
    b_mag_fld = v_mag_fld + bv_fld

    x = np.concatenate([x_mem, x_fld])
    y = np.concatenate([y_mem, y_fld])
    v_mag = np.concatenate([v_mag_true, v_mag_fld])
    b_mag = np.concatenate([b_mag_true, b_mag_fld])
    return x, y, v_mag, b_mag


def render_image(x, y, mag, size, zeropoint_flux=2.0e6, sky=300.0, gain_noise=1.0,
                  jitter=(0.0, 0.0), fwhm=3.2):
    img = np.full((size, size), sky, dtype=np.float64)
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[0:size, 0:size]
    for xi, yi, mi in zip(x, y, mag):
        xi_j, yi_j = xi + jitter[0], yi + jitter[1]
        if not (0 <= xi_j < size and 0 <= yi_j < size):
            continue
        flux = zeropoint_flux * 10 ** (-0.4 * mi)
        # only render in a small box around the star for speed
        x0, x1 = max(0, int(xi_j - 4 * sigma)), min(size, int(xi_j + 4 * sigma) + 1)
        y0, y1 = max(0, int(yi_j - 4 * sigma)), min(size, int(yi_j + 4 * sigma) + 1)
        sub_yy, sub_xx = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        g = flux / (2 * np.pi * sigma ** 2) * np.exp(-((sub_xx - xi_j) ** 2 + (sub_yy - yi_j) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += g
    # shot + read noise
    img = rng.poisson(np.clip(img, 0, None)).astype(np.float64) + rng.normal(0, gain_noise * 5, img.shape)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--n-lights", type=int, default=8)
    ap.add_argument("--n-cal", type=int, default=5)
    args = ap.parse_args()

    x, y, v_mag, b_mag = make_star_field(size=args.size)

    for filt, mags, zp in [("V", v_mag, 2.2e7), ("B", b_mag, 1.6e7)]:
        base = os.path.join(args.out, filt)
        for sub in ["lights", "darks", "flats", "bias"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

        # bias: constant + read noise
        for i in range(args.n_cal):
            b = np.full((args.size, args.size), 100.0) + rng.normal(0, 5, (args.size, args.size))
            write_fits(os.path.join(base, "bias", f"bias_{i:02d}.fits"), b)

        # darks: bias + small dark current
        for i in range(args.n_cal):
            d = np.full((args.size, args.size), 100.0 + 20.0) + rng.normal(0, 5, (args.size, args.size))
            write_fits(os.path.join(base, "darks", f"dark_{i:02d}.fits"), d)

        # flats: smooth vignette pattern + noise, bias included
        yy, xx = np.mgrid[0:args.size, 0:args.size]
        cx, cy = args.size / 2, args.size / 2
        vign = 1.0 - 0.25 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2))
        for i in range(args.n_cal):
            f = (20000 * vign) + 100.0 + rng.normal(0, 30, (args.size, args.size))
            write_fits(os.path.join(base, "flats", f"flat_{i:02d}.fits"), f)

        # lights: bias + dark + flat-modulated star field + small dithering jitter per frame
        for i in range(args.n_lights):
            jitter = (rng.normal(0, 1.5), rng.normal(0, 1.5))
            stars = render_image(x, y, mags, args.size, zeropoint_flux=zp, jitter=jitter)
            light = stars * vign + 120.0 + rng.normal(0, 5, (args.size, args.size))
            write_fits(os.path.join(base, "lights", f"light_{i:02d}.fits"), light)

        print(f"[{filt}] wrote {args.n_lights} lights, {args.n_cal} each of dark/flat/bias -> {base}/")

    print("\nDemo dataset ready. Try:")
    print(f"    python cmd_pipeline.py --data-dir {args.out} --out-dir results")


if __name__ == "__main__":
    main()
