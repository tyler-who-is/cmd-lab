#!/usr/bin/env python3
"""
cmd_gui.py — Streamlit GUI for CMD Pipeline
MaximDL-style interactive aperture photometry.

Flow:
  1. "스택 실행" → registration + stacking only
  2. 별 위치 입력 → 구경/배경고리 조절 → 한 별씩 측광
  3. "CMD 생성" → 색등급도 + 선택적 보정/ZAMS fitting
"""

import streamlit as st
import sys, os, io, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import cmd_pipeline as pipe
import plate_solve as psolver

st.set_page_config(page_title="CMD Pipeline", page_icon="⭐", layout="wide")

# ── session state ─────────────────────────────────────────────────────────
for _k, _v in [("stacks", None), ("stars", []),
                ("cmd_base", None), ("next_id", 0),
                ("plate_matches", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── helpers ───────────────────────────────────────────────────────────────

class _LogCapture(io.StringIO):
    def __init__(self, sink: list):
        super().__init__()
        self._sink = sink
    def write(self, s):
        if s.strip():
            self._sink.append(s.rstrip())
        return super().write(s)
    def flush(self):
        pass


def _measure_star(stack, x, y, r_ap, r_in, r_out):
    from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
    aper = CircularAperture([(x, y)], r=r_ap)
    annu = CircularAnnulus([(x, y)], r_in=r_in, r_out=r_out)
    sa = ApertureStats(stack, aper)
    sn = ApertureStats(stack, annu)
    bkg = float(sn.median[0])
    raw = float(sa.sum[0])
    flux = raw - bkg * aper.area
    mag = -2.5 * np.log10(flux) if flux > 0 else np.nan
    noise = np.sqrt(abs(raw) + aper.area * bkg) if bkg >= 0 else 1.0
    snr = flux / noise if noise > 0 and flux > 0 else 0.0
    return mag, flux, snr


def _find_centroid(stack, x0, y0, box=15):
    h, w = stack.shape
    r = box // 2
    ix, iy = int(round(x0)), int(round(y0))
    x1, x2 = max(0, ix - r), min(w, ix + r + 1)
    y1, y2 = max(0, iy - r), min(h, iy + r + 1)
    cut = np.nan_to_num(stack[y1:y2, x1:x2], nan=0.0)
    cut = np.clip(cut - np.median(cut), 0, None)
    total = cut.sum()
    if total <= 0:
        return x0, y0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    return float((xx * cut).sum() / total), float((yy * cut).sum() / total)


def _cutout_fig(stack, x, y, r_ap, r_in, r_out, half=30):
    h, w = stack.shape
    ix, iy = int(round(x)), int(round(y))
    x1, x2 = max(0, ix - half), min(w, ix + half + 1)
    y1, y2 = max(0, iy - half), min(h, iy + half + 1)
    cut = stack[y1:y2, x1:x2]
    fin = cut[np.isfinite(cut)]
    vm, vx = (np.percentile(fin, [1, 99.5])) if fin.size else (0, 1)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(cut, origin="lower", cmap="gray", vmin=vm, vmax=vx,
              extent=[x1, x2, y1, y2])
    for r, c, ls in [(r_ap, "lime", "-"), (r_in, "gold", "--"), (r_out, "gold", "--")]:
        ax.add_patch(plt.Circle((x, y), r, fill=False, color=c, lw=1.5, ls=ls))
    ax.plot(x, y, "+", color="red", ms=12, mew=1.5)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_title(f"({x:.1f}, {y:.1f})", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _show_fits(image, title=""):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return plt.figure()
    vmin, vmax = np.percentile(finite, [1, 99.5])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    fig.tight_layout()
    return fig


def _build_cmd_fig(df, x_label, y_label, cmd_title, zams=None, mu_used=None,
                   dist_used=None, ref_info="", calibrated=False):
    fig, ax = plt.subplots(figsize=(7, 8))
    if "candidate_member" in df.columns:
        field = df[~df.candidate_member]
        memb = df[df.candidate_member]
        ax.scatter(field.B_V, field.V, s=12, c="lightgray",
                   label=f"field / outer ({len(field)})")
        ax.scatter(memb.B_V, memb.V, s=18, c="steelblue",
                   label=f"candidate member ({len(memb)})")
    else:
        ax.scatter(df.B_V, df.V, s=18, c="steelblue")
    for _, row in df.iterrows():
        ax.annotate(f'{int(row["id"])}', (row.B_V, row.V),
                    fontsize=6, color="dimgray", ha="left",
                    xytext=(4, 2), textcoords="offset points")
    if zams is not None and mu_used is not None:
        ax.plot(zams.B_V, zams.M_V + mu_used, "r--", lw=1.5,
                label=f"ZAMS  (d≈{dist_used:.0f} pc)")
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(cmd_title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    if calibrated and ref_info:
        fig.text(0.5, 0.01, ref_info, ha="center", fontsize=7, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 1] if calibrated else None)
    return fig


def _write_explanation(path, entries, n_stars, calibrated=False,
                       zp_V=0, zp_B=0, ebv=0, rv=3.1,
                       A_V=0, A_B=0, mu_used=None, dist_used=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CMD 파이프라인 실행 과정 설명\n\n")
        for t, b in entries:
            f.write(f"## {t}\n\n{b}\n\n")
        f.write(f"## 이번 실행 요약\n\n- 수동 측정된 별: {n_stars}개\n")
        if calibrated:
            f.write(f"- zp_V={zp_V:.3f}, zp_B={zp_B:.3f}\n")
            f.write(f"- E(B-V)={ebv}, R_V={rv}  →  A_V={A_V:.3f}, A_B={A_B:.3f}\n")
            if mu_used is not None:
                f.write(f"- 거리지수 mu={mu_used:.2f}  →  거리 ≈ {dist_used:.0f} pc\n")


# ── educational texts ─────────────────────────────────────────────────────

_EXPLAIN = {
    1: ("1. 정렬 (Registration)",
        "망원경이 완벽하게 고정돼 있지 않기 때문에(트래킹 오차, 디더링 등) 사진마다 별의 "
        "위치가 픽셀 단위로 조금씩 어긋나 있습니다. 각 사진에서 별들이 이루는 패턴(삼각형 "
        "모양)을 인식해서, 기준이 되는 한 장(V필터 첫 번째 사진)의 좌표에 맞춰 나머지 "
        "사진들을 자동으로 이동/회전시킵니다 (astroalign).\n"
        "B필터 사진들도 같은 기준 프레임에 맞추므로, 나중에 같은 픽셀 좌표가 V, B "
        "두 스택에서 항상 같은 별을 가리킵니다."),

    2: ("2. 스택 (Stacking)",
        "정렬된 여러 장을 평균 내면 신호는 유지되지만 무작위 노이즈는 서로 상쇄되면서 "
        "대략 √N배 줄어듭니다. 노출 짧은 사진 여러 장을 합쳐서, 한 장으로는 안 보이던 "
        "어두운 별까지 드러나게 만드는 겁니다."),

    34: ("3–4. 수동 구경측광 (Manual Aperture Photometry)",
         "별의 위치를 직접 지정하여 측광합니다. MaximDL 등 전문 소프트웨어에서도 이와 "
         "같은 방식으로 별을 선택합니다.\n\n"
         "● 구경(aperture): 별을 감싸는 원. 이 안의 빛을 모두 더합니다.\n"
         "● 배경고리(annulus): 구경 바깥 고리. 하늘 배경 밝기를 측정합니다.\n"
         "● 기기등급 = −2.5 × log₁₀(순수 별빛)\n"
         "          = −2.5 × log₁₀(구경 총량 − 배경 × 구경 면적)\n\n"
         "구경이 너무 작으면 별빛 일부가 잘리고, 너무 크면 이웃 별의 빛이 섞입니다. "
         "배경고리에 다른 별이 걸리면 배경 추정이 왜곡되므로 고리 범위도 조절해야 합니다."),

    5: ("5. Zero-point 보정 (기기등급 → 실제등급)",
        "기기 등급을 실제 등급으로 바꾸려면 이미 실제 밝기를 아는 별이 하나 필요합니다. "
        "그 별의 (실제 등급 − 기기 등급) 차이가 zero-point이며, 이 상수 안에 망원경 구경, "
        "필터 투과율, 카메라 감도 등 장비 특성이 전부 흡수됩니다. zero-point를 모든 별에 "
        "더해주면 전체가 실제 등급계로 환산됩니다."),

    7: ("7. 주계열 비교 & 거리 추정 (ZAMS fitting)",
        "표준 주계열(ZAMS)은 어떤 색(B-V)의 별이 '원래' 얼마나 밝아야 하는지(절대등급 "
        "M_V)를 알려줍니다. 우리 성단의 별들이 같은 색이라도 ZAMS보다 어둡게 보이는 "
        "차이를 거리지수(mu)라 하고, mu = 5×log₁₀(거리pc) − 5 로 거리를 계산합니다."),
}


def _explain_extinction(ebv, A_V, A_B):
    return ("6. 성간소광 보정 (E(B−V) correction)",
            f"별빛이 먼지구름을 통과하면 파란빛이 더 많이 산란되어 실제보다 붉고 어둡게 "
            f"보입니다. E(B-V)={ebv}로 A_V={A_V:.3f}, A_B={A_B:.3f} 만큼 보정합니다.")


# ── turnoff age table ────────────────────────────────────────────────────
_TURNOFF_DATA = [
    (-0.33, "O5V",  4),    (-0.30, "B0V",  11),   (-0.28, "B1V",  16),
    (-0.22, "B2V",  32),   (-0.16, "B5V",  100),  (-0.11, "B8V",  200),
    (0.00,  "A0V",  350),  (0.16,  "A5V",  1000), (0.30,  "F0V",  2500),
    (0.44,  "F5V",  4500), (0.59,  "G0V",  8000), (0.65,  "G2V",  10000),
    (0.82,  "K0V",  17000),(1.15,  "K5V",  35000),(1.42,  "M0V",  70000),
]
_TO_BV   = np.array([t[0] for t in _TURNOFF_DATA])
_TO_SPT  = [t[1] for t in _TURNOFF_DATA]
_TO_LAGE = np.log10(np.array([t[2] for t in _TURNOFF_DATA], dtype=float))


def _turnoff_age(bv):
    log_age = float(np.interp(bv, _TO_BV, _TO_LAGE))
    idx = int(np.argmin(np.abs(_TO_BV - bv)))
    return 10 ** log_age, _TO_SPT[idx]


def _format_age(age_myr):
    if age_myr < 1000:
        return f"~{age_myr:.0f} Myr"
    return f"~{age_myr / 1000:.1f} Gyr"


def _auto_fit_cb():
    base = st.session_state.get("cmd_base")
    if base is None or not base["calibrated"]:
        return
    ebv = st.session_state.get("fit_ebv_sl", 0.0)
    mV = np.array(base["mag_V"])
    mB = np.array(base["mag_B"])
    mem = np.array(base["is_member"])
    zpV, zpB = base["zp_V"], base["zp_B"]
    A_V = 3.1 * ebv
    Vt = (mV + zpV) - A_V
    BVt = ((mB + zpB) - (mV + zpV)) - ebv
    zams = pipe.load_zams()
    mu_a, _ = pipe.fit_distance_modulus(BVt[mem], Vt[mem], zams)
    if mu_a is not None:
        st.session_state.fit_mu_sl = round(float(mu_a), 2)


# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⭐ CMD Pipeline")
    st.caption("MaximDL 스타일 수동 구경측광")

    st.header("📁 데이터")
    _data_dir = st.text_input("데이터 폴더", value="data",
                              help="V/lights/ 와 B/lights/ 가 있는 폴더")
    _out_dir = st.text_input("결과 폴더", value="results")
    data_dir = _data_dir if os.path.isabs(_data_dir) else os.path.join(SCRIPT_DIR, _data_dir)
    out_dir = _out_dir if os.path.isabs(_out_dir) else os.path.join(SCRIPT_DIR, _out_dir)

    st.header("⚙️ 스택")
    sc1, sc2 = st.columns(2)
    with sc1:
        bin_factor = st.number_input("Binning", 1, 8, 2)
    with sc2:
        max_frames = st.number_input("최대 프레임", 0, 200, 15, help="0=전부")

    st.header("🔭 구경 파라미터")
    r_ap = st.slider("구경 반지름 (px)", 2, 25, 6, help="별빛을 수집하는 원 반지름")
    r_in = st.slider("배경고리 안쪽 (px)", 4, 35, 10)
    r_out = st.slider("배경고리 바깥 (px)", 8, 50, 18)
    if r_in <= r_ap:
        st.warning("배경고리 안쪽이 구경보다 커야 합니다")
    if r_out <= r_in:
        st.warning("배경고리 바깥이 안쪽보다 커야 합니다")
    centroid_box = st.slider("중심 탐색 범위 (px)", 3, 30, 15,
                             help="입력 좌표 주변에서 별 중심을 자동 보정")
    member_frac = st.slider("멤버 반경 비율", 0.10, 0.80, 0.35, 0.05)

    st.divider()
    st.header("📤 데이터 업로드")
    st.caption("ZIP 파일 하나로 자동 분류 (폴더명 기준)")
    _zip_upload = st.file_uploader(
        "ZIP 업로드", type=["zip"], key="zip_up",
        help="V/lights, V/darks, B/lights 등 폴더 구조 그대로 압축")
    if st.button("📦 ZIP 업로드 & 분류", use_container_width=True):
        if _zip_upload is None:
            st.warning("ZIP 파일을 먼저 선택하세요")
        else:
            import zipfile
            _valid_ext = {".fit", ".fits", ".fts"}
            _filters = {"v", "b"}
            _subs = {"lights", "darks", "flats", "bias", "light", "dark", "flat"}
            _sub_map = {"light": "lights", "dark": "darks", "flat": "flats"}
            _total = 0
            _msg = []
            with zipfile.ZipFile(io.BytesIO(_zip_upload.getvalue())) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    fname = info.filename.replace("\\", "/")
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in _valid_ext:
                        continue
                    parts = [p.lower() for p in fname.split("/") if p]
                    det_filt = det_sub = None
                    for p in parts:
                        if p in _filters:
                            det_filt = p.upper()
                        if p in _subs:
                            det_sub = _sub_map.get(p, p)
                    if det_filt is None or det_sub is None:
                        continue
                    dest = os.path.join(data_dir, det_filt, det_sub)
                    os.makedirs(dest, exist_ok=True)
                    basename = os.path.basename(fname)
                    with open(os.path.join(dest, basename), "wb") as f:
                        f.write(zf.read(info.filename))
                    _total += 1
                    _msg.append(f"{det_filt}/{det_sub}")
            if _total == 0:
                st.error("분류 가능한 FITS 파일이 없습니다.\n"
                         "폴더 구조: `V/lights/`, `B/darks/` 등")
            else:
                from collections import Counter
                _counts = Counter(_msg)
                _summary = ", ".join(f"{k}: {v}장" for k, v in sorted(_counts.items()))
                st.success(f"총 {_total}장 분류 완료! {_summary}")

    with st.expander("개별 파일 업로드", expanded=False):
        st.caption("폴더별로 직접 선택")
        _fits_types = ["fit", "fits", "fts"]
        for _filt in ["V", "B"]:
            with st.expander(f"**{_filt} 필터**", expanded=False):
                _ul = {}
                for _sub in ["lights", "darks", "flats", "bias"]:
                    _ul[_sub] = st.file_uploader(
                        f"{_filt}/{_sub}", type=_fits_types,
                        accept_multiple_files=True,
                        key=f"{_filt}_{_sub}_up")
                st.session_state[f"_upload_{_filt}"] = _ul
        if st.button("📤 업로드 저장", use_container_width=True):
            _total = 0
            _msg = []
            for _filt in ["V", "B"]:
                _ul = st.session_state.get(f"_upload_{_filt}", {})
                for _sub, files in _ul.items():
                    if files:
                        dest = os.path.join(data_dir, _filt, _sub)
                        os.makedirs(dest, exist_ok=True)
                        for uf in files:
                            with open(os.path.join(dest, uf.name), "wb") as f:
                                f.write(uf.getbuffer())
                        _total += len(files)
                        _msg.append(f"{_filt}/{_sub}: {len(files)}")
            if _total == 0:
                st.warning("FITS 파일을 먼저 선택하세요")
            else:
                st.success(f"저장 완료! {', '.join(_msg)}")

    st.divider()
    if st.button("🧪 데모 데이터 생성", use_container_width=True):
        with st.spinner("데모 데이터 생성 중..."):
            import subprocess
            r = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "make_demo_data.py"),
                 "--out", os.path.join(SCRIPT_DIR, "data")],
                capture_output=True, text=True, cwd=SCRIPT_DIR)
            if r.returncode == 0:
                st.success("data/ 에 데모 데이터 생성 완료!")
            else:
                st.error(r.stderr or r.stdout)

    stack_clicked = st.button("🚀 스택 실행", type="primary",
                              use_container_width=True)

    st.divider()
    st.header("📂 CSV 불러오기")
    st.caption("이전 측정 결과 CSV로 보정 단계부터")
    csv_upload = st.file_uploader("photometry CSV", type=["csv"], key="csv_up")
    csv_load_clicked = st.button("📂 CSV로 시작", use_container_width=True)

# ── main header ───────────────────────────────────────────────────────────
st.header("CMD 파이프라인 — 수동 구경측광 모드")

# ═══════════════════════════════════════════════════════════════════════════
# STACKING
# ═══════════════════════════════════════════════════════════════════════════
if stack_clicked:
    v_lights = os.path.join(data_dir, "V", "lights")
    b_lights = os.path.join(data_dir, "B", "lights")
    ok = True
    if not os.path.isdir(v_lights):
        st.error(f"V 폴더 없음: `{v_lights}`"); ok = False
    if not os.path.isdir(b_lights):
        st.error(f"B 폴더 없음: `{b_lights}`"); ok = False
    if ok:
        v_files = pipe.find_fits(v_lights)
        b_files = pipe.find_fits(b_lights)
        if not v_files or not b_files:
            st.error("FITS 파일이 없습니다."); ok = False
    if ok:
        st.info(f"V: {len(v_files)}장 / B: {len(b_files)}장  |  bin={bin_factor}")
        log_lines: list[str] = []
        mf = None if max_frames == 0 else max_frames
        os.makedirs(out_dir, exist_ok=True)
        old_stdout = sys.stdout
        sys.stdout = _LogCapture(log_lines)
        t0 = time.time()
        progress = st.progress(0, text="스택 시작...")
        try:
            progress.progress(10, text="V 필터 정렬 + 스택 ...")
            stack_V, ref = pipe.process_filter(data_dir, "V", bin_factor, mf, None)
            progress.progress(50, text="B 필터 정렬 + 스택 ...")
            stack_B, _ = pipe.process_filter(data_dir, "B", bin_factor, mf, ref)
            from astropy.io import fits as afits
            afits.writeto(os.path.join(out_dir, "stack_V.fits"),
                          stack_V.astype(np.float32), overwrite=True)
            afits.writeto(os.path.join(out_dir, "stack_B.fits"),
                          stack_B.astype(np.float32), overwrite=True)
            elapsed = time.time() - t0
            progress.progress(100, text=f"스택 완료! ({elapsed:.1f}초)")
            st.session_state.stacks = {
                "stack_V": stack_V, "stack_B": stack_B,
                "bin_factor": bin_factor,
                "log_text": "\n".join(log_lines),
            }
            st.session_state.stars = []
            st.session_state.cmd_base = None
            st.session_state.next_id = 0
        except Exception as e:
            st.error(f"스택 오류: {e}")
            import traceback
            st.code(traceback.format_exc(), language=None)
        finally:
            sys.stdout = old_stdout

# ═══════════════════════════════════════════════════════════════════════════
# CSV LOAD
# ═══════════════════════════════════════════════════════════════════════════
if csv_load_clicked:
    if csv_upload is None:
        st.error("CSV 파일을 먼저 업로드하세요.")
    else:
        try:
            df_up = pd.read_csv(csv_upload)
            need = {"x", "y"}
            if not need.issubset(df_up.columns):
                st.error(f"CSV에 x, y 열이 필요합니다. 현재: {list(df_up.columns)}")
            else:
                star_list = []
                for _, row in df_up.iterrows():
                    sid = int(st.session_state.next_id)
                    st.session_state.next_id += 1
                    star_list.append({
                        "id": sid,
                        "x": float(row.x), "y": float(row.y),
                        "instr_V": float(row.get("instr_V", row.get("V", np.nan))),
                        "instr_B": float(row.get("instr_B",
                                    row.get("V", 0) + row.get("B_V", np.nan))),
                        "snr_V": 0.0,
                    })
                st.session_state.stars = star_list
                st.session_state.cmd_base = None
                st.success(f"CSV에서 {len(star_list)}개 별 로드 완료")
        except Exception as e:
            st.error(f"CSV 오류: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE PHOTOMETRY
# ═══════════════════════════════════════════════════════════════════════════
stacks = st.session_state.stacks
stars = st.session_state.stars

if stacks is not None:
    st.divider()
    st.subheader("🔭 구경측광")
    st.caption("이미지에서 별의 좌표를 확인 → X, Y 입력 → '별 측정' 클릭. "
               "구경/배경고리는 왼쪽 사이드바에서 조절하세요.")

    stack_V = stacks["stack_V"]
    stack_B = stacks["stack_B"]
    bfac = stacks["bin_factor"]
    h, w = stack_V.shape

    col_img, col_ctrl = st.columns([5, 3])

    # ── star chart ────────────────────────────────────────────
    with col_img:
        finite_v = stack_V[np.isfinite(stack_V)]
        vmin, vmax = np.percentile(finite_v, [1, 99.5]) if finite_v.size else (0, 1)
        img_norm = np.clip((stack_V - vmin) / (vmax - vmin), 0, 1)

        max_disp = 500
        if max(h, w) > max_disp:
            ds = int(np.ceil(max(h, w) / max_disp))
            img_small = img_norm[::ds, ::ds]
            xs = np.arange(0, w, ds, dtype=float)
            ys = np.arange(0, h, ds, dtype=float)
        else:
            img_small = img_norm
            xs = np.arange(w, dtype=float)
            ys = np.arange(h, dtype=float)

        fig_map = go.Figure()
        fig_map.add_trace(go.Heatmap(
            z=img_small, x=xs, y=ys,
            colorscale="gray", showscale=False, hoverinfo="x+y",
        ))

        # measured stars
        if stars:
            sx = [s["x"] for s in stars]
            sy = [s["y"] for s in stars]
            labels = [f'#{s["id"]}  V={s["instr_V"]:.2f}' for s in stars]
            fig_map.add_trace(go.Scatter(
                x=sx, y=sy, mode="markers+text",
                marker=dict(size=12, color="lime", symbol="circle-open",
                            line=dict(width=2, color="lime")),
                text=[f'#{s["id"]}' for s in stars],
                textposition="top right",
                textfont=dict(size=9, color="lime"),
                hovertext=labels, hoverinfo="text",
            ))
            # aperture circles for each star
            for s in stars:
                fig_map.add_shape(type="circle",
                    x0=s["x"]-r_ap, y0=s["y"]-r_ap,
                    x1=s["x"]+r_ap, y1=s["y"]+r_ap,
                    line=dict(color="lime", width=1.5))
                fig_map.add_shape(type="circle",
                    x0=s["x"]-r_in, y0=s["y"]-r_in,
                    x1=s["x"]+r_in, y1=s["y"]+r_in,
                    line=dict(color="gold", width=1, dash="dot"))
                fig_map.add_shape(type="circle",
                    x0=s["x"]-r_out, y0=s["y"]-r_out,
                    x1=s["x"]+r_out, y1=s["y"]+r_out,
                    line=dict(color="gold", width=1, dash="dot"))

        fig_map.update_layout(
            width=560, height=560,
            xaxis=dict(title="X (px)", constrain="domain", range=[-5, w+5]),
            yaxis=dict(title="Y (px)", scaleanchor="x", constrain="domain",
                       range=[-5, h+5]),
            showlegend=False,
            margin=dict(l=50, r=10, t=10, b=50),
            dragmode="zoom",
        )
        st.plotly_chart(fig_map, key="star_map")
        st.caption(f"🟢 초록 = 측정된 별  |  이미지: {w}×{h} px (bin={bfac})")

    # ── measurement controls ──────────────────────────────────
    with col_ctrl:
        st.markdown("**별 위치 입력**")
        st.caption("이미지 위에 마우스를 올리면 좌표가 보입니다")
        mc1, mc2 = st.columns(2)
        with mc1:
            inp_x = st.number_input("X", value=w // 2, min_value=0,
                                    max_value=w - 1, key="inp_x")
        with mc2:
            inp_y = st.number_input("Y", value=h // 2, min_value=0,
                                    max_value=h - 1, key="inp_y")

        # preview cutout at entered position
        fig_cut = _cutout_fig(stack_V, inp_x, inp_y, r_ap, r_in, r_out)
        st.pyplot(fig_cut)
        plt.close(fig_cut)
        st.caption("🟢 구경  🟡 배경고리  🔴 입력 위치")

        if st.button("⭐ 별 측정", type="primary", use_container_width=True):
            cx, cy = _find_centroid(stack_V, inp_x, inp_y, centroid_box)
            mV, fV, snr_V = _measure_star(stack_V, cx, cy, r_ap, r_in, r_out)
            mB, fB, snr_B = _measure_star(stack_B, cx, cy, r_ap, r_in, r_out)
            sid = st.session_state.next_id
            st.session_state.next_id += 1
            st.session_state.stars.append({
                "id": sid,
                "x": round(cx, 2), "y": round(cy, 2),
                "instr_V": round(mV, 4) if np.isfinite(mV) else np.nan,
                "instr_B": round(mB, 4) if np.isfinite(mB) else np.nan,
                "snr_V": round(snr_V, 1),
            })
            if np.isfinite(mV):
                st.success(f"별 #{sid} 측정 완료!  "
                           f"V={mV:.3f}  B={mB:.3f}  B-V={mB-mV:.3f}  "
                           f"SNR={snr_V:.0f}")
                if abs(cx - inp_x) > 0.5 or abs(cy - inp_y) > 0.5:
                    st.caption(f"중심 보정: ({inp_x}, {inp_y}) → ({cx:.1f}, {cy:.1f})")
            else:
                st.warning(f"별 #{sid}: 유효한 플럭스 없음 (구경/위치 확인)")
            st.rerun()

        # ── star list ──────────────────────────────────────────
        st.divider()
        st.markdown(f"**측정된 별 목록 ({len(stars)}개)**")
        if stars:
            df_stars = pd.DataFrame(stars)
            df_stars["B_V"] = df_stars.instr_B - df_stars.instr_V
            st.dataframe(
                df_stars[["id", "x", "y", "instr_V", "instr_B", "B_V", "snr_V"]].rename(
                    columns={"id": "#", "instr_V": "V", "instr_B": "B",
                             "B_V": "B−V", "snr_V": "SNR"}
                ),
                use_container_width=True, height=200, hide_index=True,
            )
            dc1, dc2 = st.columns([2, 1])
            with dc1:
                del_id = st.number_input("삭제할 별 #", min_value=0,
                                         max_value=max(s["id"] for s in stars),
                                         key="del_id")
            with dc2:
                st.write("")
                st.write("")
                if st.button("🗑️ 삭제"):
                    st.session_state.stars = [s for s in stars if s["id"] != del_id]
                    st.rerun()
            if st.button("🗑️ 전체 초기화", use_container_width=True):
                st.session_state.stars = []
                st.session_state.next_id = 0
                st.session_state.cmd_base = None
                st.rerun()
        else:
            st.info("아직 측정된 별이 없습니다. 위에서 좌표를 입력하고 '별 측정'을 눌러주세요.")

    # ══════════════════════════════════════════════════════════
    # CMD GENERATION
    # ══════════════════════════════════════════════════════════
    valid_stars = [s for s in stars
                   if np.isfinite(s["instr_V"]) and np.isfinite(s["instr_B"])]

    if len(valid_stars) >= 2:
        st.divider()
        st.subheader("📊 CMD 생성 & 주계열 맞추기")

        col_ref, col_gen = st.columns([3, 5])
        with col_ref:
            st.markdown("**기준별 보정 (선택사항)**")
            st.caption("실제 등급을 모르면 빈칸 → 기기등급 CMD만 생성")
            ref_ids = [s["id"] for s in valid_stars]
            ref_sel = st.selectbox("기준별 #", ref_ids, index=0,
                                   key="ref_star_sel")
            ref_star = next(s for s in valid_stars if s["id"] == ref_sel)
            st.caption(f"기기 V={ref_star['instr_V']:.3f}  "
                       f"B={ref_star['instr_B']:.3f}")

            _pm = st.session_state.get("plate_matches")
            _def_V, _def_B = 0.0, 0.0
            if _pm:
                _match_map = {m["star_id"]: m for m in _pm}
                if ref_sel in _match_map:
                    _def_V = _match_map[ref_sel]["cat_V"]
                    _def_B = _match_map[ref_sel]["cat_B"]
                    st.success(
                        f"카탈로그 매칭: V={_def_V:.3f}  B={_def_B:.3f}")

            mc1, mc2 = st.columns(2)
            with mc1:
                cal_V = st.number_input("실제 V", value=_def_V, format="%.3f",
                                        key="cal_V")
            with mc2:
                cal_B = st.number_input("실제 B", value=_def_B, format="%.3f",
                                        key="cal_B")
            do_calibrate = cal_V != 0.0 or cal_B != 0.0

            # ── plate solving ──
            with st.expander("🔭 플레이트 솔빙 (자동 표준성 검색)", expanded=False):
                st.caption("Astrometry.net + APASS 카탈로그로 "
                           "측정된 별의 실제 등급을 자동으로 찾습니다")
                _ps_key = st.text_input("API Key", value=psolver.API_KEY,
                                        key="ps_api_key")
                _ps_clicked = st.button("🔭 플레이트 솔빙 시작",
                                        use_container_width=True)
                if _ps_clicked:
                    _vfits = os.path.join(out_dir, "stack_V.fits")
                    if not os.path.isfile(_vfits):
                        st.error("stack_V.fits가 없습니다. 먼저 스택을 실행하세요.")
                    elif len(valid_stars) == 0:
                        st.error("측정된 별이 없습니다.")
                    else:
                        _xy = [[s["x"], s["y"]] for s in valid_stars]
                        _status_box = st.empty()
                        _pbar = st.progress(0, text="플레이트 솔빙 시작...")
                        _steps = {"n": 0}

                        def _ps_status(msg):
                            _steps["n"] = min(_steps["n"] + 15, 90)
                            _pbar.progress(_steps["n"], text=msg)
                            _status_box.caption(msg)

                        try:
                            wcs, matches = psolver.auto_identify(
                                _vfits, _xy,
                                api_key=_ps_key,
                                on_status=_ps_status)
                            _pbar.progress(100, text="완료!")
                            if matches:
                                _result = []
                                for m in matches:
                                    si = m["star_idx"]
                                    _result.append({
                                        "star_id": valid_stars[si]["id"],
                                        "star_x": valid_stars[si]["x"],
                                        "star_y": valid_stars[si]["y"],
                                        "cat_V": m["cat_V"],
                                        "cat_B": m["cat_B"],
                                        "sep_arcsec": m["sep_arcsec"],
                                    })
                                st.session_state.plate_matches = _result
                                st.success(
                                    f"{len(_result)}개 별 카탈로그 매칭 성공!")
                                st.rerun()
                            else:
                                st.warning("매칭된 별이 없습니다. "
                                           "별이 너무 어둡거나 시야각이 좁을 수 있습니다.")
                        except Exception as e:
                            _pbar.progress(100, text="실패")
                            st.error(f"플레이트 솔빙 오류: {e}")

                if _pm:
                    st.markdown("**매칭 결과**")
                    import pandas as _pd_ps
                    _mdf = _pd_ps.DataFrame(_pm)
                    _mdf = _mdf.rename(columns={
                        "star_id": "#", "cat_V": "V(cat)",
                        "cat_B": "B(cat)", "sep_arcsec": "거리(\")"})
                    st.dataframe(
                        _mdf[["#", "V(cat)", "B(cat)", "거리(\")"]],
                        use_container_width=True, hide_index=True)
                    st.caption("기준별을 선택하면 실제 등급이 자동 입력됩니다")

        with col_gen:
            gen_clicked = st.button("📊 CMD 생성", type="primary",
                                    use_container_width=True)

        if gen_clicked:
            os.makedirs(out_dir, exist_ok=True)
            _pos = np.array([[s["x"], s["y"]] for s in valid_stars])
            _mV = np.array([s["instr_V"] for s in valid_stars])
            _mB = np.array([s["instr_B"] for s in valid_stars])
            _ids = np.array([s["id"] for s in valid_stars])
            _cx, _cy = w / 2.0, h / 2.0
            _rd = np.sqrt((_pos[:, 0] - _cx)**2 + (_pos[:, 1] - _cy)**2)
            _mem = _rd <= (member_frac * min(h, w))

            if do_calibrate:
                _ri = next(i for i, s in enumerate(valid_stars)
                           if s["id"] == ref_sel)
                _zpV = float(cal_V - _mV[_ri])
                _zpB = float(cal_B - _mB[_ri])
            else:
                _zpV = _zpB = 0.0

            st.session_state.cmd_base = {
                "mag_V": _mV.tolist(), "mag_B": _mB.tolist(),
                "zp_V": _zpV, "zp_B": _zpB,
                "ids": _ids.tolist(),
                "positions": _pos.tolist(),
                "is_member": _mem.tolist(),
                "calibrated": do_calibrate,
                "bfac": int(bfac),
            }
            if do_calibrate:
                _zams = pipe.load_zams()
                _Vt = _mV + _zpV
                _BVt = (_mB + _zpB) - (_mV + _zpV)
                _mu_a, _ = pipe.fit_distance_modulus(
                    _BVt[_mem], _Vt[_mem], _zams)
                st.session_state.fit_mu_sl = (
                    round(float(_mu_a), 2) if _mu_a else 10.0)
                st.session_state.fit_ebv_sl = 0.0
                st.session_state.fit_to_sl = 0.0
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # INTERACTIVE ZAMS FITTING
    # ══════════════════════════════════════════════════════════
    _cb = st.session_state.get("cmd_base")
    if _cb is not None:
        st.divider()
        _mag_V = np.array(_cb["mag_V"])
        _mag_B = np.array(_cb["mag_B"])
        _zpV, _zpB = _cb["zp_V"], _cb["zp_B"]
        _ids = np.array(_cb["ids"])
        _pos = np.array(_cb["positions"])
        _mem = np.array(_cb["is_member"])
        _bfac = _cb["bfac"]

        if _cb["calibrated"]:
            st.subheader("🎯 주계열 맞추기 (Interactive ZAMS Fitting)")
            st.caption("슬라이더를 움직여 ZAMS를 데이터에 맞추고, "
                       "턴오프에서 나이를 읽으세요")

            col_sl, col_ch = st.columns([2, 5])

            with col_sl:
                st.markdown("**성간소광**")
                fit_ebv = st.slider("E(B−V)", 0.0, 2.0, 0.0, 0.01,
                                    key="fit_ebv_sl")

                st.markdown("**거리**")
                fit_mu = st.slider("거리지수 μ", 0.0, 20.0, 10.0, 0.05,
                                   key="fit_mu_sl")
                _dpc = 10 ** (1 + fit_mu / 5.0)
                st.metric("추정 거리", f"{_dpc:.0f} pc")

                st.button("🔄 자동 맞춤", on_click=_auto_fit_cb,
                          use_container_width=True)

                st.divider()
                st.markdown("**🔥 턴오프 → 나이 추정**")
                st.caption("주계열에서 별이 꺾이는 색지수를 맞추세요")
                fit_to = st.slider("턴오프 B−V", -0.35, 1.50, 0.0, 0.01,
                                   key="fit_to_sl")
                _age, _spt = _turnoff_age(fit_to)
                st.metric("추정 나이", _format_age(_age))
                st.caption(f"턴오프 분광형 ~{_spt}")

            with col_ch:
                _AV = 3.1 * fit_ebv
                _Vc = (_mag_V + _zpV) - _AV
                _BVc = ((_mag_B + _zpB) - (_mag_V + _zpV)) - fit_ebv

                _zams = pipe.load_zams()

                _df = pd.DataFrame({"id": _ids, "V": _Vc, "B_V": _BVc,
                                    "candidate_member": _mem})

                fig = go.Figure()
                _fld = _df[~_df.candidate_member]
                _mbr = _df[_df.candidate_member]

                if len(_fld) > 0:
                    fig.add_trace(go.Scatter(
                        x=_fld.B_V, y=_fld.V, mode="markers",
                        marker=dict(size=7, color="lightgray"),
                        name=f"필드 ({len(_fld)})",
                    ))
                if len(_mbr) > 0:
                    fig.add_trace(go.Scatter(
                        x=_mbr.B_V, y=_mbr.V, mode="markers+text",
                        marker=dict(size=9, color="steelblue"),
                        text=[str(int(i)) for i in _mbr.id],
                        textposition="top right",
                        textfont=dict(size=8, color="dimgray"),
                        name=f"멤버 후보 ({len(_mbr)})",
                    ))

                fig.add_trace(go.Scatter(
                    x=_zams.B_V, y=_zams.M_V + fit_mu,
                    mode="lines",
                    line=dict(color="red", dash="dash", width=2),
                    name=f"ZAMS (d≈{_dpc:.0f} pc)",
                ))

                _to_mv = float(np.interp(
                    fit_to, _zams.B_V.values, _zams.M_V.values))
                fig.add_vline(x=fit_to,
                              line=dict(color="orange", width=1.5,
                                        dash="dot"))
                fig.add_trace(go.Scatter(
                    x=[fit_to], y=[_to_mv + fit_mu],
                    mode="markers",
                    marker=dict(size=14, color="orange", symbol="star"),
                    name=f"턴오프 ({_format_age(_age)})",
                ))

                fig.update_layout(
                    xaxis_title="B − V (dereddened)",
                    yaxis_title="V (dereddened)",
                    yaxis=dict(autorange="reversed"),
                    height=650,
                    legend=dict(x=0.98, y=0.02, xanchor="right",
                                yanchor="bottom"),
                    margin=dict(l=50, r=20, t=30, b=50),
                )
                st.plotly_chart(fig, use_container_width=True,
                                key="cmd_fit_chart")

            # ── downloads ────────────────────────────────────
            _df_save = pd.DataFrame({
                "id": _ids,
                "x": _pos[:, 0] * _bfac, "y": _pos[:, 1] * _bfac,
                "instr_V": _mag_V, "instr_B": _mag_B,
                "V": _Vc, "B_V": _BVc,
                "candidate_member": _mem,
            })
            dc1, dc2 = st.columns(2)
            with dc1:
                st.download_button("💾 photometry.csv",
                                   _df_save.to_csv(index=False),
                                   "photometry.csv", "text/csv",
                                   use_container_width=True)
            with dc2:
                _summary = (
                    f"# ZAMS Fitting 결과\n\n"
                    f"E(B-V) = {fit_ebv:.3f}\n"
                    f"μ = {fit_mu:.2f}\n"
                    f"거리 ≈ {_dpc:.0f} pc\n"
                    f"턴오프 B-V = {fit_to:.2f}\n"
                    f"추정 나이 ≈ {_format_age(_age)}\n"
                    f"턴오프 분광형 ~{_spt}\n")
                st.download_button("💾 결과 요약.txt", _summary,
                                   "fitting_result.txt", "text/plain",
                                   use_container_width=True)

        else:
            # ── uncalibrated instrumental CMD ────────────────
            st.subheader("📊 CMD (기기등급)")
            _BVc = _mag_B - _mag_V
            _df = pd.DataFrame({"id": _ids, "V": _mag_V, "B_V": _BVc,
                                "candidate_member": _mem})
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=_df.B_V, y=_df.V, mode="markers+text",
                marker=dict(size=9, color="steelblue"),
                text=[str(int(i)) for i in _df.id],
                textposition="top right",
                textfont=dict(size=8),
            ))
            fig.update_layout(
                xaxis_title="instrumental B − V",
                yaxis_title="instrumental V",
                yaxis=dict(autorange="reversed"), height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("💾 photometry.csv",
                               _df.to_csv(index=False),
                               "photometry.csv", "text/csv")

    # ── stack images & explanations ──────────────────────────
    st.divider()
    tab_img, tab_exp = st.tabs(["🖼️ 스택 이미지", "📜 설명/로그"])
    with tab_img:
        ic1, ic2 = st.columns(2)
        with ic1:
            fv = _show_fits(stack_V, "V-band stack")
            st.pyplot(fv); plt.close(fv)
        with ic2:
            fb = _show_fits(stack_B, "B-band stack")
            st.pyplot(fb); plt.close(fb)
    with tab_exp:
        for _t, _b in [_EXPLAIN[1], _EXPLAIN[2], _EXPLAIN[34]]:
            with st.expander(_t, expanded=False):
                st.write(_b)
        if _cb and _cb["calibrated"]:
            with st.expander(_EXPLAIN[5][0], expanded=False):
                st.write(_EXPLAIN[5][1])
            with st.expander(_EXPLAIN[7][0], expanded=False):
                st.write(_EXPLAIN[7][1])
        st.subheader("스택 로그")
        st.code(stacks.get("log_text", ""), language=None)

elif stacks is None and not stack_clicked:
    st.markdown(
        """
        ### 사용 방법

        1. **스택 실행** → V/B 이미지 자동 정렬 + 합산
        2. 스택 이미지에서 **별 좌표 확인** → X, Y 입력 → **별 측정**
        3. 구경/배경고리를 사이드바에서 조절하며 반복
        4. 별이 2개 이상이면 **CMD 생성** 가능
        5. 기준별의 실제 등급을 입력 → **주계열 맞추기** (거리 + 나이 추정)

        데모 데이터가 없으면 **🧪 데모 데이터 생성** 버튼을 먼저 눌러주세요.
        """
    )
