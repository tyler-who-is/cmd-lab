#!/usr/bin/env python3
"""
cmd_pipeline.py -- One-button B/V photometry -> Color-Magnitude Diagram pipeline
for an open-cluster HR-diagram classroom activity.

WHAT IT DOES (fully automatic, no clicking through a GUI):
  1. Loads already-calibrated B/V light frames (bias/dark/flat correction is
     assumed already done -- see "CALIBRATION" note below if it isn't).
  2. Registers (aligns) every frame -- of BOTH filters -- onto one common
     reference frame using star-pattern matching (astroalign), so the V
     stack and the B stack end up on the exact same pixel grid. This means
     "the star at pixel (x,y)" is the same physical star in both stacks,
     so no separate B<->V star cross-matching step is needed later.
  3. Stacks (average) each filter into a master image.
  4. Detects every star automatically in the V stack (DAOStarFinder).
  5. Does aperture photometry for every detected star on BOTH the V and B
     stacks at the same pixel coordinates -> instrumental magnitudes.
  6. OPTIONAL, if you give it one star of known brightness (--ref-... flags
     below): turns instrumental magnitudes into real apparent B, V
     magnitudes (zero-point calibration), corrects for interstellar
     reddening/extinction (--ebv), and overlays the reference (Mamajek)
     zero-age main sequence on the plot so you can visually compare your
     cluster's main sequence + turnoff against it.
  7. Flags stars close to the field center as "candidate cluster members"
     (crude radius cut -- see README for the caveat).
  8. Writes photometry.csv and cmd_plot.png (Color-Magnitude Diagram).

CALIBRATION (bias/dark/flat):
  By default this script assumes you hand it ALREADY light-calibrated FITS
  files (bias/dark/flat already applied, e.g. by Siril). Just put them in
  data/V/lights/ and data/B/lights/ -- nothing else is needed.
  If you'd rather let this script also do bias/dark/flat correction itself,
  just additionally populate data/V/darks|flats|bias/ and data/B/darks|flats|bias/
  and it will use them automatically; if those folders are empty/missing it
  skips that step with a warning, so both workflows "just work".

MAGNITUDE CALIBRATION (turning instrumental mags into real B, V mags):
  Give the script ONE star in your field whose real brightness you already
  know (e.g. from SIMBAD/a catalog), and it derives the zero-point for you:

    --ref-x 1234 --ref-y 987          pixel position of that star, read off
                                       the FIRST V-filter light frame in your
                                       data folder (native/unbinned pixels --
                                       the script rescales for --bin itself)
    --ref-V 8.03 --ref-B 8.87         its known apparent B, V magnitudes
        -- OR, if you only know its absolute magnitude + distance --
    --ref-Vabs 0.5 --ref-Babs 0.9 --ref-dist-pc 1900

  Add interstellar extinction correction (recommended -- ask your literature
  source, e.g. for M11 the published mean reddening is roughly E(B-V)=0.7):

    --ebv 0.74          [--rv 3.1 is the standard default, rarely needs changing]

  Add a known distance to overlay the reference main sequence at the right
  apparent brightness (otherwise the script AUTO-FITS the distance modulus
  by sliding the main sequence until it best matches your candidate-member
  stars, and reports the implied distance -- this is literally the
  "main-sequence fitting" technique):

    --distance-pc 1900

  If you skip all --ref-* flags, the script just plots instrumental
  (uncalibrated) magnitudes with no main-sequence overlay, same as before.

USAGE:
    python cmd_pipeline.py
    python cmd_pipeline.py --bin 2 --max-frames 15
    python cmd_pipeline.py --ref-x 1234 --ref-y 987 --ref-V 8.03 --ref-B 8.87 --ebv 0.74

See README.md for full instructions and a worked example.
"""

import argparse
import time
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
import astroalign as aa
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FITS_EXTS = (".fit", ".fits", ".fts")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZAMS_PATH = os.path.join(SCRIPT_DIR, "zams_mamajek.csv")

EXPLAIN = True          # set False by --quiet
EXPLAIN_LOG = []        # collects (title, body) for results/explanation.md


def explain(title, body):
    """Print (and log for explanation.md) a short teaching note about the
    step that's about to run. This is a classroom tool -- the whole point
    is that students can see *why* each step exists, not just a progress
    bar."""
    EXPLAIN_LOG.append((title, body))
    if not EXPLAIN:
        return
    print(f"\n\033[36m--- [{title}] ---\033[0m")
    for line in body.strip().split("\n"):
        print(f"    {line}")


# --------------------------------------------------------------------------
# Utility: I/O
# --------------------------------------------------------------------------

def find_fits(folder):
    files = []
    if not os.path.isdir(folder):
        return files
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(FITS_EXTS):
            files.append(os.path.join(folder, f))
    return files


def load_fits_data(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data


def bin_image(img, factor):
    """Simple integer block-mean downsampling."""
    if factor is None or factor <= 1:
        return img
    h, w = img.shape
    h2, w2 = h - (h % factor), w - (w % factor)
    img = img[:h2, :w2]
    return img.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# Calibration (bias/dark/flat) -- optional, auto-skips if no cal frames given
# --------------------------------------------------------------------------

def build_master(files, label, bin_factor):
    if not files:
        return None
    stack = [bin_image(load_fits_data(f), bin_factor) for f in files]
    master = np.median(np.array(stack), axis=0)
    print(f"    master {label}: {len(files)} frame(s) -> shape {master.shape}")
    return master


def build_master_flat(files, master_bias, bin_factor):
    if not files:
        return None
    stack = []
    for f in files:
        d = bin_image(load_fits_data(f), bin_factor)
        if master_bias is not None:
            d = d - master_bias
        stack.append(d)
    flat = np.median(np.array(stack), axis=0)
    flat = flat / np.median(flat)
    flat[flat <= 0] = 1.0
    print(f"    master flat: {len(files)} frame(s) -> shape {flat.shape}")
    return flat


def calibrate_light(raw, master_bias, master_dark, master_flat):
    d = raw.copy()
    if master_bias is not None:
        d = d - master_bias
    if master_dark is not None:
        d = d - (master_dark - (master_bias if master_bias is not None else 0))
    if master_flat is not None:
        d = d / master_flat
    return d


# --------------------------------------------------------------------------
# Registration + stacking
# --------------------------------------------------------------------------

def register_and_accumulate(calibrated_frames, reference, acc, count_acc, label):
    n_ok, n_fail = 0, 0
    for i, frame in enumerate(calibrated_frames):
        try:
            registered = frame if reference is None else aa.register(frame, reference)[0]
            acc += registered
            count_acc += 1
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"    [{label}] frame {i}: registration failed, skipped ({e})")
    print(f"    [{label}] registered {n_ok} frame(s), {n_fail} failed/skipped")
    return n_ok


# --------------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------------

def detect_stars(image, fwhm=4.0, nsigma=5.0):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)
    sources = finder(image - median)
    if sources is None:
        return np.empty((0, 2))
    return np.transpose((sources["xcentroid"], sources["ycentroid"]))


def aperture_instrumental_mag(image, positions, r_ap=6.0, r_in=10.0, r_out=16.0):
    apertures = CircularAperture(positions, r=r_ap)
    annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    bkg_mean = ApertureStats(image, annulus, sigma_clip=SigmaClip(sigma=3.0)).mean
    raw_sum = ApertureStats(image, apertures).sum
    net_flux = raw_sum - bkg_mean * apertures.area
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(net_flux)
    return mag, net_flux


# --------------------------------------------------------------------------
# Magnitude calibration (zero point, reddening, ZAMS overlay)
# --------------------------------------------------------------------------

def load_zams(path=ZAMS_PATH):
    df = pd.read_csv(path)
    return df.sort_values("B_V").reset_index(drop=True)


def fit_distance_modulus(bv0, v0, zams, mu_grid=None):
    """Grid-search the distance modulus mu that best slides the ZAMS
    (M_V) onto the observed dereddened (B-V, V) points. Returns best mu
    and the implied distance in pc."""
    if mu_grid is None:
        mu_grid = np.arange(4.0, 16.0, 0.02)
    zams_bv = zams["B_V"].values
    zams_mv = zams["M_V"].values
    valid = np.isfinite(bv0) & np.isfinite(v0)
    bv0, v0 = bv0[valid], v0[valid]
    if len(bv0) < 3:
        return None, None
    # for each star, nearest ZAMS point in color -> predicted M_V
    order = np.argsort(zams_bv)
    zbv, zmv = zams_bv[order], zams_mv[order]
    pred_mv = np.interp(bv0, zbv, zmv, left=zmv[0], right=zmv[-1])
    best_mu, best_cost = None, np.inf
    for mu in mu_grid:
        resid = v0 - (pred_mv + mu)
        cost = np.median(np.abs(resid))  # robust to outliers/contamination
        if cost < best_cost:
            best_cost, best_mu = cost, mu
    dist_pc = 10 ** (1 + best_mu / 5.0)
    return best_mu, dist_pc


def nearest_star_index(positions, x, y, max_dist):
    d = np.sqrt((positions[:, 0] - x) ** 2 + (positions[:, 1] - y) ** 2)
    i = np.argmin(d)
    if d[i] > max_dist:
        return None, d[i]
    return i, d[i]


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def process_filter(data_dir, filt, bin_factor, max_frames, global_ref):
    print(f"\n[{filt}] loading frames ...")
    base = os.path.join(data_dir, filt)
    light_files = find_fits(os.path.join(base, "lights"))
    dark_files = find_fits(os.path.join(base, "darks"))
    flat_files = find_fits(os.path.join(base, "flats"))
    bias_files = find_fits(os.path.join(base, "bias"))

    if not light_files:
        raise SystemExit(f"No light frames found in {base}/lights -- check --data-dir.")

    if max_frames:
        light_files = light_files[:max_frames]

    if bias_files or dark_files or flat_files:
        master_bias = build_master(bias_files, "bias", bin_factor) if bias_files else None
        master_dark = build_master(dark_files, "dark", bin_factor) if dark_files else None
        master_flat = build_master_flat(flat_files, master_bias, bin_factor) if flat_files else None
    else:
        print("    no darks/flats/bias folders found -- assuming lights are already calibrated")
        master_bias = master_dark = master_flat = None

    calibrated = []
    for f in light_files:
        raw = bin_image(load_fits_data(f), bin_factor)
        calibrated.append(calibrate_light(raw, master_bias, master_dark, master_flat))
    print(f"    loaded {len(calibrated)} light frame(s), shape {calibrated[0].shape}")

    if global_ref is None:
        global_ref = calibrated[0]
        print("    using this filter's first frame as the GLOBAL reference frame")

    acc = np.zeros_like(global_ref, dtype=np.float64)
    count = register_and_accumulate(calibrated, global_ref, acc, 0, filt)
    if count == 0:
        raise SystemExit(f"[{filt}] every frame failed to register -- aborting.")
    return acc / count, global_ref


def main():
    ap = argparse.ArgumentParser(description="One-button B/V CMD photometry pipeline.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--bin", type=int, default=2, help="block-mean downsample factor (default 2, use 1 for full-res)")
    ap.add_argument("--max-frames", type=int, default=15, help="max light frames per filter, for speed (0 = all)")
    ap.add_argument("--fwhm", type=float, default=4.0, help="expected stellar FWHM in (binned) pixels")
    ap.add_argument("--nsigma", type=float, default=6.0, help="detection threshold in sky-noise sigma")
    ap.add_argument("--member-frac", type=float, default=0.35, help="fraction of half-frame radius = 'candidate member'")

    g = ap.add_argument_group("magnitude calibration (all optional)")
    g.add_argument("--ref-x", type=float, help="reference star x pixel (native/unbinned coords, from the 1st V light frame)")
    g.add_argument("--ref-y", type=float, help="reference star y pixel (native/unbinned coords)")
    g.add_argument("--ref-match-radius", type=float, default=30.0, help="max snap distance in native px (default 30)")
    g.add_argument("--ref-V", type=float, help="reference star's known apparent V magnitude")
    g.add_argument("--ref-B", type=float, help="reference star's known apparent B magnitude")
    g.add_argument("--ref-Vabs", type=float, help="reference star's known absolute M_V (use with --ref-dist-pc instead of --ref-V)")
    g.add_argument("--ref-Babs", type=float, help="reference star's known absolute M_B (use with --ref-dist-pc instead of --ref-B)")
    g.add_argument("--ref-dist-pc", type=float, help="distance to the reference star in pc (needed if using --ref-Vabs/--ref-Babs)")
    g.add_argument("--ebv", type=float, default=0.0, help="interstellar reddening E(B-V) mag (default 0 = no correction)")
    g.add_argument("--rv", type=float, default=3.1, help="total-to-selective extinction ratio R_V (default 3.1)")
    g.add_argument("--distance-pc", type=float, help="known cluster distance in pc, to place the ZAMS overlay (omit to auto-fit)")
    ap.add_argument("--quiet", action="store_true", help="suppress the teaching explanations printed at each step")

    args = ap.parse_args()

    global EXPLAIN
    EXPLAIN = not args.quiet

    max_frames = None if args.max_frames == 0 else args.max_frames
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("CMD PIPELINE -- open cluster B/V photometry")
    print("=" * 60)

    explain(
        "1. 정렬 (Registration)",
        """
        망원경이 완벽하게 고정돼 있지 않기 때문에(트래킹 오차, 디더링 등) 사진마다 별의
        위치가 픽셀 단위로 조금씩 어긋나 있습니다. 지금부터 각 사진에서 별들이 이루는
        패턴(삼각형 모양)을 인식해서, 기준이 되는 한 장(V필터 첫 번째 사진)의 좌표에 맞춰
        나머지 사진들을 자동으로 이동/회전시킵니다 (astroalign 라이브러리 사용).
        중요한 점: B필터 사진들도 전부 이 '같은' 기준 프레임에 맞춰서 정렬하기 때문에,
        나중에 특정 픽셀 좌표의 별이 V, B 두 스택에서 항상 같은 별을 가리키게 됩니다.
        """
    )
    stack_V, ref = process_filter(args.data_dir, "V", args.bin, max_frames, None)
    stack_B, _ = process_filter(args.data_dir, "B", args.bin, max_frames, ref)

    explain(
        "2. 스택 (Stacking)",
        """
        한 장의 사진에는 진짜 별빛(신호)과 무작위로 흔들리는 밝기 잡음(노이즈)이 섞여
        있습니다. 정렬된 여러 장을 평균 내면 신호는 그대로 유지되지만, 무작위 노이즈는
        서로 상쇄되면서 대략 sqrt(N)배 줄어듭니다 (N = 합친 장수). 그래서 노출 짧은
        사진 여러 장을 합쳐서, 한 장으로는 노이즈에 묻혀 안 보이던 어두운 별까지
        드러나게 만드는 겁니다.
        """
    )
    fits.writeto(os.path.join(args.out_dir, "stack_V.fits"), stack_V.astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(args.out_dir, "stack_B.fits"), stack_B.astype(np.float32), overwrite=True)
    print(f"\nSaved stack_V.fits / stack_B.fits to {args.out_dir}/")

    explain(
        "3. 별 자동 검출 (Star detection)",
        """
        스택된 이미지에서, 주변 하늘 배경보다 통계적으로 확실히(--nsigma 배 표준편차
        이상) 밝게 뭉쳐있는 점들을 찾습니다 (DAOStarFinder). 그 점의 밝기 분포 모양이
        실제 별의 모습(PSF, Point Spread Function)과 비슷한지도 확인해서, 우주선(cosmic
        ray) 히트나 핫픽셀 같은 가짜 신호는 걸러냅니다.
        """
    )
    positions = detect_stars(stack_V, fwhm=args.fwhm, nsigma=args.nsigma)
    print(f"  detected {len(positions)} star(s)")
    if len(positions) == 0:
        raise SystemExit("No stars detected -- try lowering --nsigma or check your data.")

    explain(
        "4. 조리개 측광 (Aperture photometry)",
        """
        검출된 각 별의 위치에 작은 원(조리개, aperture)을 그려서 그 안의 모든 빛을
        더합니다. 다만 이 값에는 별빛 + 하늘 배경광이 섞여 있으므로, 조리개 바로 바깥의
        고리 모양 영역(annulus)에서 하늘 배경의 평균 밝기를 따로 측정해 빼줍니다. 이렇게
        구한 순수 별빛의 양(net flux)에 m = -2.5 × log10(flux) 공식을 적용하면 '기기
        등급(instrumental magnitude)'이 나옵니다 -- 로그 스케일이라 등급이 5 작아질
        때마다 실제 밝기는 100배 밝아집니다. 이 값은 아직 우리 망원경/카메라 기준의
        상대적인 값이라, 논문이나 카탈로그의 등급과 바로 비교할 수는 없습니다.
        """
    )
    mag_V, _ = aperture_instrumental_mag(stack_V, positions)
    mag_B, _ = aperture_instrumental_mag(stack_B, positions)

    h, w = stack_V.shape
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2)
    is_member = r <= (args.member_frac * min(h, w))

    # ---- optional magnitude calibration ----
    calibrated_mode = False
    zp_V = zp_B = 0.0
    A_V = A_B = 0.0
    ref_info = ""
    have_ref_mag = (args.ref_V is not None and args.ref_B is not None) or \
                   (args.ref_Vabs is not None and args.ref_Babs is not None and args.ref_dist_pc is not None)
    if args.ref_x is not None and args.ref_y is not None and have_ref_mag:
        explain(
            "5. Zero-point 보정 (기기등급 → 실제등급)",
            """
            기기 등급을 실제 등급(다른 사람의 관측/카탈로그와 비교 가능한 값)으로 바꾸려면
            이미 실제 밝기를 알고 있는 별이 하나 필요합니다. 그 별의 '실제 등급 − 기기
            등급' 차이를 zero-point라고 부르는데, 이 하나의 상수 안에 망원경 조리개 크기,
            필터 투과율, 카메라 감도 같은 우리 장비만의 특성이 전부 흡수되어 있습니다.
            이 zero-point를 우리가 측정한 모든 별에 똑같이 더해주면, 전체 별이 한꺼번에
            실제 등급계로 환산됩니다.
            """
        )
        bx, by = args.ref_x / args.bin, args.ref_y / args.bin
        idx, dist = nearest_star_index(positions, bx, by, args.ref_match_radius / args.bin)
        if idx is None:
            print(f"\n[calibration] WARNING: no detected star within {args.ref_match_radius}px of "
                  f"({args.ref_x},{args.ref_y}) -- skipping magnitude calibration, plotting instrumental mags only.")
        else:
            if args.ref_V is not None:
                ref_V_app, ref_B_app = args.ref_V, args.ref_B
            else:
                mu_ref = 5 * np.log10(args.ref_dist_pc) - 5
                ref_V_app, ref_B_app = args.ref_Vabs + mu_ref, args.ref_Babs + mu_ref
            zp_V = ref_V_app - mag_V[idx]
            zp_B = ref_B_app - mag_B[idx]
            A_V = args.rv * args.ebv
            A_B = A_V + args.ebv
            calibrated_mode = True
            ref_info = (f"ref star: matched detected star id={idx} at ({positions[idx,0]*args.bin:.0f},"
                        f"{positions[idx,1]*args.bin:.0f}) px, {dist*args.bin:.1f}px from requested position\n"
                        f"  -> zero points: zp_V={zp_V:.3f}  zp_B={zp_B:.3f}   (A_V={A_V:.3f}, A_B={A_B:.3f} from E(B-V)={args.ebv})")
            print(f"\n[calibration] {ref_info}")
    elif args.ref_x is not None or args.ref_V is not None or args.ref_Vabs is not None:
        print("\n[calibration] WARNING: incomplete --ref-* arguments given (need --ref-x, --ref-y, and either "
              "--ref-V/--ref-B or --ref-Vabs/--ref-Babs[+--ref-dist-pc]) -- skipping calibration.")

    if calibrated_mode:
        if args.ebv:
            explain(
                "6. 성간소광/적색화 보정 (E(B-V) correction)",
                f"""
                별빛이 우리 은하 안의 먼지구름을 통과하면, 파란빛이 붉은빛보다 더 많이
                산란되기 때문에 실제보다 더 붉고(색이 커지고) 더 어둡게 관측됩니다. 이
                효과를 성간적색화/성간소광이라고 부릅니다. E(B-V)=(관측된 B-V) − (원래
                B-V)는 색이 얼마나 더 붉어졌는지를 나타내고, A_V = R_V × E(B-V)
                (R_V≈3.1, 표준값) 공식으로 V등급이 얼마나 더 어두워졌는지도 계산됩니다.
                지금 입력한 E(B-V)={args.ebv}로 A_V={A_V:.3f}, A_B={A_B:.3f}등급만큼을
                실제 색/밝기에서 빼서, 먼지가 없었다면 보였을 원래 모습으로 복원합니다.
                """
            )
        cal_V = mag_V + zp_V
        cal_B = mag_B + zp_B
        V_col = cal_V - A_V           # dereddened apparent V
        BV_col = (cal_B - cal_V) - args.ebv   # dereddened B-V
        y_label = "apparent V (dereddened)"
        x_label = "B - V  (dereddened)"
        title = "Color-Magnitude Diagram (calibrated, dereddened)"
    else:
        V_col = mag_V
        BV_col = mag_B - mag_V
        y_label = "instrumental V (fainter →)"
        x_label = "instrumental B - V"
        title = "Color-Magnitude Diagram (instrumental, uncalibrated)"

    good = np.isfinite(V_col) & np.isfinite(BV_col)
    print(f"\n  {good.sum()} / {len(positions)} star(s) have valid B and V photometry (rest saturated/negative flux, dropped)")

    df = pd.DataFrame({
        "id": np.arange(len(positions))[good],
        "x": positions[good, 0] * args.bin,
        "y": positions[good, 1] * args.bin,
        "instr_V": mag_V[good],
        "instr_B": mag_B[good],
        "V": V_col[good],
        "B_V": BV_col[good],
        "radius_px": r[good],
        "candidate_member": is_member[good],
    })
    csv_path = os.path.join(args.out_dir, "photometry.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}  ({len(df)} rows)")

    # ---- ZAMS overlay + optional distance-modulus fit ----
    zams = None
    mu_used = dist_used = None
    if calibrated_mode:
        explain(
            "7. 주계열 비교 & 거리 추정 (ZAMS fitting)",
            """
            표준 주계열(Zero-Age Main Sequence)은 거리를 이미 정확히 아는 근처 별들로
            만든 기준선이라, 어떤 색(B-V)을 가진 별이 '원래' 얼마나 밝아야 하는지(절대
            등급 M_V)를 알려줍니다. 우리 성단의 별들은 훨씬 멀리 있어서, 같은 색이라도
            실제로는 주계열과 똑같이 밝은 별이 겉보기엔 더 어둡게 보입니다. 이 '겉보기
            등급 − 절대등급' 차이를 거리지수(distance modulus, mu)라고 하고,
            mu = 5×log10(거리pc) − 5 관계로 거리를 계산할 수 있습니다. 우리 데이터를
            주계열에 수직으로 맞춰볼 때(성단 중심부 candidate_member 별들만 사용) 가장
            잘 맞는 mu를 찾는 것이 '주계열 맞추기(main-sequence fitting)' 기법이며, 그
            결과로 나온 거리 위에서 관측된 주계열이 표준 주계열과 벌어지는 지점(턴오프)의
            위치를 보고 성단의 나이를 추정할 수 있습니다 (나이가 많을수록 턴오프가 더
            어둡고 붉은 쪽으로 내려옵니다).
            """
        )
        zams = load_zams()
        memb_mask = df.candidate_member
        if args.distance_pc:
            mu_used = 5 * np.log10(args.distance_pc) - 5
            dist_used = args.distance_pc
            print(f"\n[ZAMS] using given distance = {dist_used:.0f} pc  (mu = {mu_used:.2f})")
        else:
            mu_used, dist_used = fit_distance_modulus(df.B_V[memb_mask].values, df.V[memb_mask].values, zams)
            if mu_used is not None:
                print(f"\n[ZAMS] auto-fit distance modulus mu = {mu_used:.2f}  ->  distance ≈ {dist_used:.0f} pc "
                      f"(fit using {memb_mask.sum()} candidate-member stars; treat as a rough estimate)")
            else:
                print("\n[ZAMS] not enough candidate-member stars to auto-fit a distance modulus -- overlay skipped.")

    # --- Plot CMD ---
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    field = df[~df.candidate_member]
    memb = df[df.candidate_member]
    ax.scatter(field.B_V, field.V, s=10, c="lightgray", label=f"field / outer ({len(field)})")
    ax.scatter(memb.B_V, memb.V, s=16, c="steelblue", label=f"candidate member ({len(memb)})")

    if zams is not None and mu_used is not None:
        ax.plot(zams.B_V, zams.M_V + mu_used, "r--", lw=1.5,
                label=f"Mamajek ZAMS  (d≈{dist_used:.0f} pc)")

    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    if calibrated_mode:
        note = ref_info.split("\n")[-1].strip()
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 1] if calibrated_mode else None)
    png_path = os.path.join(args.out_dir, "cmd_plot.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")

    # ---- write a self-contained teaching summary (great for a lab report) ----
    md_path = os.path.join(args.out_dir, "explanation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CMD 파이프라인 실행 과정 설명\n\n")
        f.write("이 문서는 `cmd_pipeline.py`가 이번 실행에서 실제로 거친 각 단계와, "
                "그 단계가 왜 필요한지를 자동으로 정리한 것입니다. 실습 보고서에 그대로 "
                "붙여 넣어도 됩니다.\n\n")
        for title, body in EXPLAIN_LOG:
            f.write(f"## {title}\n\n")
            f.write(" ".join(line.strip() for line in body.strip().split("\n")) + "\n\n")
        f.write("## 이번 실행에서 실제로 계산된 값\n\n")
        f.write(f"- 검출된 별: {len(positions)}개 (그중 B/V 둘 다 유효한 측광값을 가진 별: {good.sum()}개)\n")
        f.write(f"- 성단 후보 멤버: {int(is_member[good].sum())}개 / 필드(비멤버) 후보: {int((~is_member[good]).sum())}개\n")
        if calibrated_mode:
            f.write(f"- {ref_info.splitlines()[0]}\n")
            f.write(f"- zero-point: zp_V={zp_V:.3f}, zp_B={zp_B:.3f}\n")
            f.write(f"- E(B-V)={args.ebv}, R_V={args.rv}  ->  A_V={A_V:.3f}, A_B={A_B:.3f}\n")
            if mu_used is not None:
                f.write(f"- 거리지수 mu={mu_used:.2f}  ->  거리 ≈ {dist_used:.0f} pc "
                        f"{'(직접 입력값)' if args.distance_pc else '(주계열 자동 맞춤 추정치)'}\n")
        else:
            f.write("- 기준별 정보가 없어서 기기등급만 계산했습니다 (실제 등급/주계열 비교 없음).\n")
    print(f"Saved {md_path}  (수업/보고서용 설명 문서)")

    print(f"\nDone in {time.time() - t0:.1f} s.")
    print("=" * 60)


if __name__ == "__main__":
    main()
