import os
import cv2
import time
import subprocess as sp
import math
import argparse
import numpy as np
import json
# try:
#     from .ffprobe import ffprobe  # when imported as a package module
# except ImportError:  # when run as a standalone script
#     from ffprobe import ffprobe
from typing import Optional, Dict, Any
import hashlib

def ffprobe(filename):
    """get metadata by using ffprobe
    Checks the output of ffprobe on the desired video
    file. MetaData is then parsed into a dictionary.
    Parameters
    ----------
    filename : string
        Path to the video file
    Returns
    -------
    metaDict : dict
       Dictionary containing all header-based information
       about the passed-in source video.
    """
    # Call ffprobe for streams
    streams_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-select_streams", "v:0",
        "-print_format", "json",
        filename
    ]
    packets_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_packets",
        "-select_streams", "v:0",
        "-print_format", "json",
        filename
    ]
    result_streams = sp.run(
        streams_cmd,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=True
    )
    result_packets = sp.run(
        packets_cmd,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=True
    )
    viddict = json.loads(result_streams.stdout)
    packets = json.loads(result_packets.stdout)
    return viddict, packets



# ---------------- YUV plane parsers ----------------
def _split_yuv420_planes(buf: bytes, H: int, W: int, layout: str):
    """Return Y (H,W), U (H/2,W/2), V (H/2,W/2) for layout in {i420,yv12,nv12,nv21}."""
    nY = H*W
    nUV = (H//2)*(W//2)
    arr = np.frombuffer(buf, dtype=np.uint8)
    if layout in ("i420","yv12"):
        Y = arr[:nY].reshape(H, W)
        UV = arr[nY:]
        # planar U and V (each nUV)
        U_planar, V_planar = (UV[:nUV], UV[nUV:]) if layout=="i420" else (UV[nUV:], UV[:nUV])
        U = U_planar.reshape(H//2, W//2)
        V = V_planar.reshape(H//2, W//2)
        return Y, U, V
    elif layout in ("nv12","nv21"):
        Y = arr[:nY].reshape(H, W)
        UVint = arr[nY:].reshape(H//2, W)  # interleaved per row: UVUV or VUVU
        U = np.empty((H//2, W//2), dtype=np.uint8)
        V = np.empty((H//2, W//2), dtype=np.uint8)
        if layout == "nv12":  # UVUV...
            U[:] = UVint[:, 0::2]
            V[:] = UVint[:, 1::2]
        else:                  # nv21: VUVU...
            V[:] = UVint[:, 0::2]
            U[:] = UVint[:, 1::2]
        return Y, U, V
    else:
        raise ValueError(layout)


# ---------------- manual YUV->BGR with matrix/range ----------------
def _upsample_uv(U, V, H, W):
    # nearest-neighbor 2x upsample
    U_up = cv2.resize(U, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    V_up = cv2.resize(V, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return U_up, V_up

def _yuv_to_bgr_manual(buf: bytes, H: int, W: int, layout: str, matrix: str, rng: str):
    """
    matrix: 'bt601' or 'bt709'
    rng   : 'tv' or 'full'
    """
    Y, U, V = _split_yuv420_planes(buf, H, W, layout)
    Yf = Y.astype(np.float32)
    Uf, Vf = U.astype(np.float32), V.astype(np.float32)
    U_up, V_up = _upsample_uv(Uf, Vf, H, W)

    # Center Cb/Cr
    Cb = U_up - 128.0
    Cr = V_up - 128.0

    if matrix == "bt601":
        if rng == "tv":
            # ITU-R BT.601 limited range
            C = Yf - 16.0
            R = 1.164383 * C + 1.596027 * Cr
            G = 1.164383 * C - 0.391762 * Cb - 0.812968 * Cr
            B = 1.164383 * C + 2.017232 * Cb
        else: # full
            R = Yf + 1.402000 * Cr
            G = Yf - 0.344136 * Cb - 0.714136 * Cr
            B = Yf + 1.772000 * Cb
    elif matrix == "bt709":
        if rng == "tv":
            C = Yf - 16.0
            R = 1.164383 * C + 1.792741 * Cr
            G = 1.164383 * C - 0.213249 * Cb - 0.532909 * Cr
            B = 1.164383 * C + 2.112402 * Cb
        else: # full
            # full-range近似（常用系数）
            R = Yf + 1.5748 * Cr
            G = Yf - 0.1873 * Cb - 0.4681 * Cr
            B = Yf + 1.8556 * Cb
    else:
        raise ValueError(matrix)

    B = np.clip(B, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    R = np.clip(R, 0, 255).astype(np.uint8)
    bgr = np.dstack([B, G, R])
    return bgr, Y  # also return the original Y for scoring


def _bgr_to_y_estimate(bgr: np.ndarray, matrix: str, rng: str):
    B = bgr[:,:,0].astype(np.float32)
    G = bgr[:,:,1].astype(np.float32)
    R = bgr[:,:,2].astype(np.float32)
    if matrix == "bt601":
        kr, kb = 0.299, 0.114
    else:  # bt709
        kr, kb = 0.2126, 0.0722
    kg = 1.0 - kr - kb
    if rng == "tv":
        Y = 16.0 + 219.0 * (kr*R + kg*G + kb*B)/255.0
    else:
        Y = (kr*R + kg*G + kb*B)
    return np.clip(Y, 0, 255).astype(np.uint8)

def _auto_pick_color(yuv_buf: bytes, H: int, W: int):
    """ 穷举 {i420,yv12,nv12,nv21}×{bt601,bt709}×{tv,full}，选重建Y与原始Y最接近的组合 """
    layouts = ["i420", "yv12", "nv12", "nv21"]
    matrices = ["bt601", "bt709"]
    ranges = ["tv", "full"]
    best = None
    best_mse = 1e18
    picked_bgr = None
    for lay in layouts:
        try:
            for mat in matrices:
                for rng in ranges:
                    bgr, Yorig = _yuv_to_bgr_manual(yuv_buf, H, W, lay, mat, rng)
                    Yest = _bgr_to_y_estimate(bgr, mat, rng)
                    mse = float(np.mean((Yest.astype(np.int16)-Yorig.astype(np.int16))**2))
                    if mse < best_mse:
                        best_mse = mse
                        best = (lay, mat, rng)
                        picked_bgr = bgr
        except Exception:
            continue
    if best is None:
        # 回退
        return "i420", "bt601", "tv", _yuv_to_bgr_manual(yuv_buf, H, W, "i420", "bt601", "tv")[0]
    return best[0], best[1], best[2], picked_bgr

# ---------------- robust HEVC frame reader ----------------
class RobustHevcStream:
    """
    Decoder stdout layout (order固定, 尺寸可能不同):
      YUV420 | MVX_L0 | MVY_L0 | MVX_L1 | MVY_L1 | REF_OFF_L0 | REF_OFF_L1 | SIZE | (padding) | META | YUV420_RES
    自适应:
      * MV元素字节: int16 vs int32
      * SIZE长度: H/8*W/8 vs 与单个MV平面等长的字节数
    """
    def __init__(self, video, parallel=1, hevc_bin=None):
        self.video = video
        self.parallel = str(parallel)
        self.hevc_bin = hevc_bin or os.environ.get('HEVC_FEAT_DECODER', 'hevc')
        if not (os.path.isfile(self.hevc_bin) and os.access(self.hevc_bin, os.X_OK)):
            raise FileNotFoundError(f"HEVC binary not found/executable: {self.hevc_bin}")
        vinfo, _ = ffprobe(video)
        self.W  = int(vinfo.get("width", 0))
        self.H  = int(vinfo.get("height", 0))
        self.nf = int(vinfo.get("nb_frames", "0") or 0)
        # sizes
        self.Y = self.W * self.H
        self.U = (self.W>>1)*(self.H>>1)
        self.V = self.U
        self.yuv_bytes = self.Y + self.U + self.V
        self.mv_elems  = (self.W>>2)*(self.H>>2)
        self.size_elems = (self.W>>3)*(self.H>>3)
        self.mv_plane_b16 = self.mv_elems * 2
        self.mv_plane_b32 = self.mv_elems * 4
        self.off_plane_b  = self.mv_elems
        self.meta_bytes   = self.Y >> 2
        self.res_bytes    = self.yuv_bytes
        self._devnull = open(os.devnull, 'wb')
        self.proc = sp.Popen([self.hevc_bin, "-i", self.video, "-p", self.parallel],
                             stdin=sp.PIPE, stdout=sp.PIPE, stderr=self._devnull)
        self.buf = bytearray()

    def close(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.close()
            except Exception: pass
            try: self.proc.stdout.close()
            except Exception: pass
            self.proc.terminate()
        self.proc = None
        if hasattr(self, '_devnull') and self._devnull:
            try: self._devnull.close()
            except Exception: pass
            self._devnull = None

    def _read_exact(self, n):
        out = bytearray()
        if self.buf:
            take = min(n, len(self.buf))
            out += self.buf[:take]; del self.buf[:take]; n -= take
        while n > 0:
            chunk = self.proc.stdout.read(n)
            if not chunk:
                raise RuntimeError(f"Short read need {n} more bytes")
            out += chunk; n -= len(chunk)
        return bytes(out)

    def _prefetch(self, n):
        missing = n - len(self.buf)
        while missing > 0:
            chunk = self.proc.stdout.read(missing)
            if not chunk:
                raise RuntimeError(f"Short read while prefetch {missing}")
            self.buf += chunk; missing -= len(chunk)

    def read_one(self):
        """
        Read a single frame from the HEVC feature decoder, following the same
        binary layout as HevcFeatureReader._read_frame_data:

          YUV420 | MVX_L0 | MVY_L0 | MVX_L1 | MVY_L1 |
          REF_OFF_L0 | REF_OFF_L1 | SIZE | (padding) | META | YUV420_RES

        All MV planes are int16; SIZE plane is uint8 with length (H/8)*(W/8),
        stored in a buffer of length equal to one MV plane.
        """
        # 1) Raw YUV420 for this frame
        yuv = self._read_exact(self.yuv_bytes)

        # 2) Sizes consistent with HevcFeatureReader._read_frame_data
        pvY_size = self.Y
        pvU_size = self.U
        pvV_size = self.V

        pvMV_size = self.mv_plane_b16          # (W/4 * H/4) * 2 bytes, int16
        pvOFF_size = self.off_plane_b         # (W/4 * H/4) bytes, uint8
        pvSize_size = self.size_elems         # (W/8 * H/8) bytes, uint8

        # Padding between SIZE and META (can be zero)
        pvOffset = (3 * self.W * self.H >> 2) - (pvMV_size * 5 + pvOFF_size * 2)
        if pvOffset < 0:
            raise RuntimeError(f"Computed negative pvOffset={pvOffset}, check geometry.")

        # 3) Motion vectors and reference offsets (L0/L1), all planes at quarter-res
        mvxL0_b = self._read_exact(pvMV_size)
        mvyL0_b = self._read_exact(pvMV_size)
        mvxL1_b = self._read_exact(pvMV_size)
        mvyL1_b = self._read_exact(pvMV_size)

        ref0_b  = self._read_exact(pvOFF_size)
        ref1_b  = self._read_exact(pvOFF_size)

        # 4) SIZE plane stored in a buffer with the same length as one MV plane;
        #    only the first pvSize_size entries are meaningful.
        size_raw_b = self._read_exact(pvMV_size)

        # 5) Skip padding region (if any) between SIZE and META.
        if pvOffset > 0:
            _ = self._read_exact(pvOffset)

        # 6) META block and residual YUV420
        meta = self._read_exact(self.meta_bytes)
        if not (len(meta) >= 2 and meta[0] == 4 and meta[1] == 2):
            raise RuntimeError("META header not [4,2]")

        resid = self._read_exact(self.res_bytes)

        # 7) Convert raw buffers to numpy arrays with proper shapes/dtypes
        H4, W4 = self.H >> 2, self.W >> 2
        H8, W8 = self.H >> 3, self.W >> 3

        mvxL0 = np.frombuffer(mvxL0_b, dtype=np.int16).reshape(H4, W4)
        mvyL0 = np.frombuffer(mvyL0_b, dtype=np.int16).reshape(H4, W4)
        mvxL1 = np.frombuffer(mvxL1_b, dtype=np.int16).reshape(H4, W4)
        mvyL1 = np.frombuffer(mvyL1_b, dtype=np.int16).reshape(H4, W4)

        ref0  = np.frombuffer(ref0_b, dtype=np.uint8).reshape(H4, W4)
        ref1  = np.frombuffer(ref1_b, dtype=np.uint8).reshape(H4, W4)

        size_map = np.frombuffer(size_raw_b, dtype=np.uint8)[:pvSize_size].reshape(H8, W8)

        return {
            "yuv": yuv,
            "resid": resid,
            "meta": meta,
            "mvxL0": mvxL0,
            "mvyL0": mvyL0,
            "mvxL1": mvxL1,
            "mvyL1": mvyL1,
            "ref0": ref0,
            "ref1": ref1,
            "size": size_map,
        }


def viz_mv_to_hsv(mvx: np.ndarray, mvy: np.ndarray) -> np.ndarray:
    """
    Visualize motion vectors as an HSV image converted to BGR (uint8).
    mvx, mvy: arrays of shape (H, W) or (H/4, W/4).
    Returns a BGR uint8 image of shape (H, W, 3).
    Hue encodes direction (angle), Value encodes magnitude (normalized by 95th percentile), Saturation=255.
    If input is (H/4,W/4), we upsample to (H,W) with nearest.
    """
    assert mvx.shape == mvy.shape
    H, W = mvx.shape
    # If looks like quarter-res, upsample to full size using nearest
    full_H = getattr(viz_mv_to_hsv, "_full_H", None)
    full_W = getattr(viz_mv_to_hsv, "_full_W", None)
    if full_H is not None and full_W is not None and (H*4 == full_H and W*4 == full_W):
        mvx_u = cv2.resize(mvx, (full_W, full_H), interpolation=cv2.INTER_NEAREST)
        mvy_u = cv2.resize(mvy, (full_W, full_H), interpolation=cv2.INTER_NEAREST)
    else:
        mvx_u, mvy_u = mvx, mvy

    ang = np.arctan2(-mvy_u, mvx_u)  # y-down image coords
    ang = (ang + np.pi) / (2*np.pi)  # [0,1]
    mag = np.sqrt(mvx_u**2 + mvy_u**2)
    s = np.percentile(mag, 95) if np.isfinite(mag).all() else 1.0
    s = max(s, 1e-6)
    mag = np.clip(mag / s, 0, 1)

    hsv = np.zeros((mvx_u.shape[0], mvx_u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 179).astype(np.uint8)   # OpenCV Hue in [0,179]
    hsv[..., 1] = 255
    hsv[..., 2] = (mag * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def viz_residual(res: np.ndarray, signed: bool = True) -> np.ndarray:
    """
    Visualize residual (Y or RGB). For Y, expects shape (H,W) or (1,H,W).
    If signed=True and dtype is uint8 in [0,255], center at 128 and show divergence with COLORMAP_TURBO.
    If signed=False, show magnitude via COLORMAP_TURBO.
    Returns BGR uint8 image.
    """
    if res.ndim == 3 and res.shape[0] in (1,3):
        # CHW -> HWC
        if res.shape[0] == 1:
            res = res[0]
        else:
            res = res.transpose(1,2,0)
    if res.ndim == 2:
        img = res.astype(np.float32)
        if signed:
            img = (img - 128.0)  # center
            a = np.percentile(np.abs(img), 95)
            a = max(a, 1.0)
            img = (img + a) / (2*a)  # [-a,a] -> [0,1]
        else:
            a = np.percentile(img, 95)
            a = max(a, 1.0)
            img = np.clip(img / a, 0, 1)
        vis = (img * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    elif res.ndim == 3 and res.shape[2] == 3:
        # Assume BGR or RGB in 0..255
        vis = res.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected residual shape for viz: {res.shape}")
    return vis

_HEVC_FEAT_DECODER = os.environ.get('HEVC_FEAT_DECODER', 'hevc')

_FFMPEG_SUPPORTED_DECODERS = [ext.encode() for ext in [
    ".mp4", ".mkv", ".mov", ".hevc", ".h265", ".265"
]]

def _parse_fraction(frac: Optional[str]) -> float:
    if not frac or frac == "0/0":
        return 0.0
    try:
        if "/" in frac:
            a, b = frac.split("/")
            a = float(a.strip()); b = float(b.strip())
            return 0.0 if b == 0 else a / b
        return float(frac)
    except Exception:
        return 0.0


class HevcFeatureReader:
    """Reads frame features using HevcFeatureReader
    Return quadtree structure, yuv data, residual, raw motion vectors.
    """

    def __init__(self, filename, nb_frames, n_parallel):
        # General information
        _, self.extension = os.path.splitext(filename)
        if not os.path.exists(filename):
            print(filename, " not exist.")
        viddict_raw, packets_raw = ffprobe(filename)

        # Helper: get first-present key
        def _get(d, *keys):
            for k in keys:
                if isinstance(d, dict) and k in d:
                    return d[k]
            return None

        # Normalize stream dict: support JSON ("streams") and XML-like ("stream") outputs
        streams = _get(viddict_raw, "streams", "stream")
        if isinstance(streams, list):
            viddict = streams[0] if streams else {}
        else:
            viddict = streams or {}

        # Normalize packets list
        packets_list = _get(packets_raw, "packets", "packet") or []
        if isinstance(packets_list, dict):
            packets_list = [packets_list]

        # Extract PTS for bitstream order (best-effort)
        packets_pts = []
        for p in packets_list:
            v = _get(p, "pts", "@pts")
            if v is None:
                continue
            try:
                packets_pts.append(int(v))
            except Exception:
                try:
                    packets_pts.append(int(float(v)))
                except Exception:
                    pass
        if not packets_pts:
            packets_pts = list(range(len(packets_list)))

        self.viddict = viddict
        self.bitstream_pts_order = np.argsort(packets_pts)
        self.decode_order = np.argsort(self.bitstream_pts_order)

        self.bpp = -1  # bits per pixel
        self.pix_fmt = _get(viddict, "pix_fmt", "@pix_fmt")
        if nb_frames is not None:
            self.nb_frames = nb_frames
        else:
            nbf = _get(viddict, "nb_frames", "@nb_frames")
            try:
                self.nb_frames = int(nbf)
            except Exception:
                self.nb_frames = len(packets_list) if packets_list else 0

        self.width = int(_get(viddict, "width", "@width"))
        self.height = int(_get(viddict, "height", "@height"))

        cw = _get(viddict, "coded_width", "@coded_width")
        ch = _get(viddict, "coded_height", "@coded_height")
        self.coded_width = int(cw) if cw is not None else self.width
        self.coded_height = int(ch) if ch is not None else self.height

        self.ctu_width = math.ceil(self.width / 64.0)
        self.ctu_height = math.ceil(self.height / 64.0)
        self.nb_ctus = self.ctu_width * self.ctu_height

        if self.pix_fmt not in ["yuv420p"]:
            # print(self.pix_fmt)
            # print(filename)
            raise NameError("Expect a yuv420p input.")

        assert str.encode(self.extension).lower() in _FFMPEG_SUPPORTED_DECODERS, (
                "Unknown decoder extension: " + self.extension.lower()
        )

        # FPS / time base estimation (best-effort from ffprobe)
        avg_fps = _get(viddict, "avg_frame_rate", "@avg_frame_rate")
        r_fps   = _get(viddict, "r_frame_rate", "@r_frame_rate")
        self.fps = _parse_fraction(avg_fps) or _parse_fraction(r_fps) or 0.0
        self.time_base = (1.0 / self.fps) if self.fps > 0 else None

        # Frame/GOP counters
        self._frame_index = 0
        self._gop_id = -1
        self._pos_in_gop = -1

        self._filename = filename

        # Verify external HEVC feature decoder binary exists and is executable
        if not os.path.isfile(_HEVC_FEAT_DECODER) or not os.access(_HEVC_FEAT_DECODER, os.X_OK):
            raise FileNotFoundError(
                f"HEVC feature decoder not found or not executable at '{_HEVC_FEAT_DECODER}'.\n"
                f"Set env HEVC_FEAT_DECODER to the correct binary path."
            )

        self.DEVNULL = open(os.devnull, "wb")

        # Create process
        self._parallel = str(n_parallel)
        cmd = [_HEVC_FEAT_DECODER] + ["-i", self._filename] + ["-p", self._parallel]
        # print(" ".join(cmd))
        self._proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=self.DEVNULL)

        # place-holder for last meta
        self._lastmeta: Dict[str, Any] = {}

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            # self._proc.stderr.close()
            self._terminate(0.2)
        self._proc = None

    def _terminate(self, timeout=1.0):
        """Terminate the sub process."""
        # Check
        if self._proc is None:  # pragma: no cover
            return  # no process
        if self._proc.poll() is not None:
            return  # process already dead
        # Terminate process
        self._proc.terminate()
        # Wait for it to close (but do not get stuck)
        etime = time.time() + timeout
        while time.time() < etime:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _read_frame_data(self):
        self.pvY_size = self.width * self.height
        self.pvU_size = (self.width >> 1) * (self.height >> 1)
        self.pvV_size = (self.width >> 1) * (self.height >> 1)

        pvMV_size = (self.width >> 2) * (self.height >> 2) * 2
        pvOFF_size = (self.width >> 2) * (self.height >> 2)
        pvSize_size = (self.width >> 3) * (self.height >> 3)
        pvOffset = (3 * self.width * self.height >> 2) - (
                pvMV_size * 5 + pvOFF_size * 2
        )
        assert self._proc is not None

        # Helper to ensure full buffer is read; if not, raise early with helpful msg
        def _read_exact(num_bytes: int) -> bytes:
            buf = self._proc.stdout.read(num_bytes)
            if buf is None or len(buf) != num_bytes:
                self._terminate()
                raise RuntimeError(
                    f"Short read from decoder. Expected {num_bytes} bytes, got {0 if buf is None else len(buf)}."
                )
            return buf

        try:

            arr_YUV420 = np.frombuffer(
                _read_exact(self.pvY_size + self.pvU_size + self.pvV_size),
                dtype=np.uint8,
            )
            arr_MVX_L0 = np.frombuffer(
                _read_exact(pvMV_size), dtype=np.int16
            )
            arr_MVY_L0 = np.frombuffer(
                _read_exact(pvMV_size), dtype=np.int16
            )
            arr_MVX_L1 = np.frombuffer(
                _read_exact(pvMV_size), dtype=np.int16
            )
            arr_MVY_L1 = np.frombuffer(
                _read_exact(pvMV_size), dtype=np.int16
            )

            arr_REF_OFF_L0 = np.frombuffer(
                _read_exact(pvOFF_size), dtype=np.uint8
            )
            arr_REF_OFF_L1 = np.frombuffer(
                _read_exact(pvOFF_size), dtype=np.uint8
            )
            arr_Size = np.frombuffer(_read_exact(pvMV_size), dtype=np.uint8)[
                       :pvSize_size
                       ]
            _ = _read_exact(pvOffset)
            arr_meta = np.frombuffer(
                _read_exact(self.pvY_size >> 2), dtype=np.uint8
            )
            arr_YUV420_residual = np.frombuffer(
                _read_exact(self.pvY_size + self.pvU_size + self.pvV_size),
                dtype=np.uint8,
            )
            assert arr_meta[0] == 4 and arr_meta[1] == 2
            assert len(arr_meta) == self.pvY_size >> 2

        except Exception as e:
            print(e)
            self._terminate()
            raise RuntimeError(
                "Failed to decode video. video information: ", self.viddict
            )

        return (
            arr_meta,
            arr_YUV420,
            arr_MVX_L0,
            arr_MVY_L0,
            arr_MVX_L1,
            arr_MVY_L1,
            arr_REF_OFF_L0,
            arr_REF_OFF_L1,
            arr_Size,
            arr_YUV420_residual,
        )

    def _is_i_like(self, frame_type: int, ref_off_L0: np.ndarray, ref_off_L1: np.ndarray,
                   mv_x_L0: np.ndarray, mv_y_L0: np.ndarray,
                   mv_x_L1: np.ndarray, mv_y_L1: np.ndarray) -> bool:
        """
        Heuristic: consider a frame 'I-like' if (a) decoder marks it as I/IDR (common encoders use 0 or 1),
        OR (b) no references + near-zero MVs.
        L0 only (single-ref, no B-frames).
        """
        STRICT_I = int(os.environ.get('UMT_HEVC_STRICT_I', '0')) != 0
        try:
            # Use env var to override I types, default '0'
            _itypes_env = os.environ.get('UMT_HEVC_I_TYPES', '0')
            try:
                _itypes = {int(x) for x in _itypes_env.replace(' ', '').split(',') if x != ''}
            except Exception:
                _itypes = {0}
            if int(frame_type) in _itypes:
                return True
        except Exception:
            pass
        if STRICT_I:
            # In strict mode, do NOT use the fallback; rely only on frame_type flag.
            return False
        # Fallback: no references and tiny motion (use L0 only)
        no_ref = (ref_off_L0.max() == 0)
        mv_max = max(np.abs(mv_x_L0).max(), np.abs(mv_y_L0).max())
        return bool(no_ref and mv_max == 0)

    @staticmethod
    def _i_cache_key_from_y(y_plane: np.ndarray) -> str:
        """
        Build a stable cache key for the I-frame content (downsample to 8x8 then md5).
        """
        # ensure uint8 2D array
        y_small = cv2.resize(y_plane, (8, 8), interpolation=cv2.INTER_AREA)
        m = hashlib.md5(y_small.tobytes())
        return m.hexdigest()

    def _readFrame(self):
        (
            arr_meta,
            arr_YUV420,
            arr_MVX_L0,
            arr_MVY_L0,
            arr_MVX_L1,
            arr_MVY_L1,
            arr_REF_OFF_L0,
            arr_REF_OFF_L1,
            arr_Size,
            arr_YUV420_residual,
        ) = self._read_frame_data()

        frame_type = arr_meta[2]
        quadtree_stru = arr_meta[1024: 1024 + self.nb_ctus * 12]

        all_yuv_data = arr_YUV420.reshape(self.height + (self.height >> 1), self.width)
        all_yuv_data_residual = arr_YUV420_residual.reshape(
            self.height + (self.height >> 1), self.width
        )

        if int(os.environ.get('UMT_HEVC_Y_ONLY', '1')) != 0:
            y = all_yuv_data[:self.height, :self.width]
            y_res = all_yuv_data_residual[:self.height, :self.width]
            rgb = y
            residual = y_res
        else:
            rgb = cv2.cvtColor(all_yuv_data, cv2.COLOR_YUV420p2BGR)
            residual = cv2.cvtColor(all_yuv_data_residual, cv2.COLOR_YUV420p2BGR)

        # Optionally suppress I-frame RGB payload (student side takes I-RGB via decord)
        _SUPPRESS_I = int(os.environ.get('UMT_HEVC_SUPPRESS_I_RGB', '0')) != 0

        mv_x_L0 = arr_MVX_L0.reshape(self.height >> 2, self.width >> 2)
        mv_y_L0 = arr_MVY_L0.reshape(self.height >> 2, self.width >> 2)
        mv_x_L1 = arr_MVX_L1.reshape(self.height >> 2, self.width >> 2)
        mv_y_L1 = arr_MVY_L1.reshape(self.height >> 2, self.width >> 2)
        ref_off_L0 = arr_REF_OFF_L0.reshape(self.height >> 2, self.width >> 2)
        ref_off_L1 = arr_REF_OFF_L1.reshape(self.height >> 2, self.width >> 2)

        size = arr_Size.reshape(self.height >> 3, self.width >> 3)

        # ---- meta info (GOP / timestamp / cache key) ----
        is_i_frame = self._is_i_like(frame_type, ref_off_L0, ref_off_L1,
                                     mv_x_L0, mv_y_L0, mv_x_L1, mv_y_L1)
        # Suppress I-frame RGB if requested
        if _SUPPRESS_I and bool(is_i_frame):
            # Replace RGB with a zero image of the same shape to avoid expensive usage downstream.
            # Keep residual as-is (student still needs residual guidance from HEVC path).
            if isinstance(rgb, np.ndarray) and rgb.ndim == 2:
                # gray path -> expand to 3 channels of zeros for consistent shape
                rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            elif isinstance(rgb, np.ndarray) and rgb.ndim == 3:
                rgb = np.zeros_like(rgb)
            else:
                rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Maintain GOP id and position within GOP (pos starts at 0 for the first frame of a GOP)
        if self._frame_index == 0:
            # Start first GOP at the first frame regardless of type, to keep indices stable
            self._gop_id = 0
            self._pos_in_gop = 0
            if is_i_frame:
                # it's an I at the head of GOP 0 (common case)
                pass
        else:
            if is_i_frame:
                # New GOP begins
                self._gop_id += 1 if self._gop_id >= 0 else 0
                self._pos_in_gop = 0
            else:
                # Continue within current GOP (if _pos_in_gop not yet initialized, start from 0)
                self._pos_in_gop = (0 if self._pos_in_gop < 0 else self._pos_in_gop + 1)

        # timestamp estimation
        ts = (self._frame_index * self.time_base) if self.time_base is not None else None

        # build a human-readable frame type string (best-effort)
        try:
            _ft = int(frame_type)
            # More explicit labels: common mapping 0=IDR, 1=CRA(IRAP), 2=P, 3=B
            _ft_map = {0: "IDR", 1: "CRA", 2: "P", 3: "B"}
            ft_str = _ft_map.get(_ft, str(_ft))
        except Exception:
            ft_str = "NA"

        # optional I cache key (Y-only) to help upper layers deduplicate compute
        if int(os.environ.get('UMT_HEVC_Y_ONLY', '1')) != 0:
            i_cache_key = self._i_cache_key_from_y(rgb) if is_i_frame else None
        else:
            # if using BGR, derive key from its Y channel approximation
            y_tmp = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)[:,:,0] if (isinstance(rgb, np.ndarray) and rgb.ndim==3) else rgb
            i_cache_key = self._i_cache_key_from_y(y_tmp) if is_i_frame else None

        # Per-frame tiny hash (8x8 downsample of Y channel) for alignment debugging
        try:
            if int(os.environ.get('UMT_HEVC_Y_ONLY', '1')) != 0:
                _y_small = cv2.resize(all_yuv_data[:self.height, :self.width], (8, 8), interpolation=cv2.INTER_AREA)
            else:
                _y_small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)[:,:,0], (8, 8), interpolation=cv2.INTER_AREA)
            _frame_hash = hashlib.md5(_y_small.tobytes()).hexdigest()
        except Exception:
            _frame_hash = None

        self._lastmeta = {
            "frame_index": int(self._frame_index),
            "gop_id": int(self._gop_id),
            "is_i_frame": bool(is_i_frame),
            "frame_type": int(frame_type) if np.isscalar(frame_type) or isinstance(frame_type, (int, np.integer)) else int(frame_type.item()),
            "frame_type_str": ft_str,
            "timestamp": ts,
            "i_cache_key": i_cache_key,
            "width": int(self.width),
            "height": int(self.height),
            "gop_pos": [int(self._gop_id), int(max(0, self._pos_in_gop))],
            "frame_hash": _frame_hash,
            "i_rgb_suppressed": bool(_SUPPRESS_I and is_i_frame),
        }
        self._frame_index += 1

        self._lastread = (
            frame_type,
            quadtree_stru,
            rgb,
            mv_x_L0,
            mv_y_L0,
            mv_x_L1,
            mv_y_L1,
            ref_off_L0,
            ref_off_L1,
            size,
            residual,
        )

        return self._lastread

    def nextFrame(self):
        """Yields hevc features using a generator"""
        for i in range(self.nb_frames):
            yield self._readFrame()

    def nextFrameEx(self):
        """Like nextFrame(), but yields a (tuple, meta_dict). Keeps backward compatibility."""
        for i in range(self.nb_frames):
            data = self._readFrame()
            yield data, dict(self._lastmeta)

    def getFrameNums(self):
        return self.nb_frames

    def getShape(self):
        return self.width, self.height

    def getDecodeOrder(self):
        return self.decode_order


    # ----------- Fast-path: extract first GOP as tensors for student model ----------- #
    def _upsample_mv_to_hw(self, mv):
        """
        Nearest-neighbor upsample of (H/4, W/4) MV array to (H, W).
        Args:
            mv: np.ndarray shape (H/4, W/4)
        Returns:
            np.ndarray shape (H, W)
        """
        H4, W4 = mv.shape
        H, W = self.height, self.width
        return cv2.resize(mv, (W, H), interpolation=cv2.INTER_NEAREST)

    def _normalize_mv_and_res(self, mvx, mvy, res, mode='percentile', pct=95, signed_res=True):
        """
        Normalize MV and residual arrays for model input.
        Args:
            mvx, mvy: np.ndarray [N, H, W] (int16 or float)
            res: np.ndarray [N, C, H, W] (uint8 or float)
            mode: 'percentile', 'fixed', or 'log1p'
            pct: percentile for scaling
            signed_res: map residual to [-1,1] (if True) or [0,1] (if False)
        Returns:
            mvx_norm, mvy_norm, res_norm: normalized arrays
        """
        # Normalize MV
        mvx = mvx.astype(np.float32)
        mvy = mvy.astype(np.float32)
        if mode == 'percentile':
            s = np.percentile(np.abs(np.concatenate([mvx.flatten(), mvy.flatten()])), pct)
            s = max(s, 1.0)
            mvx_norm = np.clip(mvx / s, -1.0, 1.0)
            mvy_norm = np.clip(mvy / s, -1.0, 1.0)
            mvx_norm = (mvx_norm + 1) / 2  # [0,1]
            mvy_norm = (mvy_norm + 1) / 2
        elif mode == 'fixed':
            # Assume 127 is reasonable max
            mvx_norm = np.clip(mvx / 127.0, -1.0, 1.0)
            mvy_norm = np.clip(mvy / 127.0, -1.0, 1.0)
            mvx_norm = (mvx_norm + 1) / 2
            mvy_norm = (mvy_norm + 1) / 2
        elif mode == 'log1p':
            mvx_norm = np.log1p(np.abs(mvx)) * np.sign(mvx)
            mvy_norm = np.log1p(np.abs(mvy)) * np.sign(mvy)
            # rescale to [-1,1] then [0,1]
            mx = max(np.abs(mvx_norm).max(), 1e-3)
            my = max(np.abs(mvy_norm).max(), 1e-3)
            mvx_norm = np.clip(mvx_norm / mx, -1.0, 1.0)
            mvy_norm = np.clip(mvy_norm / my, -1.0, 1.0)
            mvx_norm = (mvx_norm + 1) / 2
            mvy_norm = (mvy_norm + 1) / 2
        else:
            raise ValueError(f"Unknown norm_mode {mode}")

        # Normalize residual
        res = res.astype(np.float32)
        if signed_res:
            # Assume residual is centered around 0 and ranges roughly [-255, 255].
            max_val = np.percentile(np.abs(res), pct) if mode == 'percentile' else 255.0
            max_val = max(max_val, 1.0)
            res_norm = np.clip(res / max_val, -1.0, 1.0)
        else:
            # If treating as magnitude only, map absolute values to [0,1]
            max_val = np.percentile(res, pct) if mode == 'percentile' else 255.0
            max_val = max(max_val, 1.0)
            res_norm = np.clip(res / max_val, 0.0, 1.0)
        return mvx_norm, mvy_norm, res_norm


    def read_first_gop_tensors(self, gop_size=12, need_i_rgb=True, mv_level='L0',
                               upsample_mv=True, norm_mode='percentile', mv_pct=95, signed_res=True,
                               across_gops=True, collect_mv_from_all=False, across_gop=None):
        """
        Quickly read the first GOP (gop_id=0) and return stacked numpy tensors for student input.
        Returns:
          tensors: dict with keys:
            'I':  np.ndarray of shape [Ti, Ci, H, W]  (Ti<=gop_size; Ci=1(gray) or 3)
            'MV': np.ndarray of shape [Tp, 2, H, W]    (2 = x,y)
            'R':  np.ndarray of shape [Tp, Cr, H, W]   (Cr=1 if Y-only else 3)
          metas: list of per-frame meta dicts (length Ti+Tp up to gop_size)
        Notes:
          - If `across_gops` is True (default), we take the first `gop_size` frames in decode order and IGNORE GOP boundaries.
            If False, we stop at the first frame whose `gop_id` changes from 0 (i.e., only the first GOP is considered).
          - If `collect_mv_from_all` is True, we also collect MV/RES for frames marked as I-like; otherwise仅收集非 I 帧的 MV/RES。
          - Frames beyond `gop_size` are ignored; decoding stops early for speed.
        # NOTE: single-ref stream; only L0 MV is used.
        """
        # Backward-compat: allow caller to pass across_gop (singular)
        if across_gop is not None:
            across_gops = bool(across_gop)
        # Storage
        i_frames = []
        i_metas = []
        p_mv_x = []
        p_mv_y = []
        p_res = []
        p_metas = []
        H, W = self.height, self.width
        y_only = int(os.environ.get('UMT_HEVC_Y_ONLY', '1')) != 0
        n_col = 1 if y_only else 3
        n_res_col = 1 if y_only else 3
        n_i_col = 1 if y_only else 3
        n_p_col = n_res_col
        n_mv_ch = 2
        n_frames = 0
        DEBUG = int(os.environ.get('UMT_HEVC_DEBUG', '0')) != 0
        _dbg_count = 0       # limit per-frame add logs
        _scan_count = 0      # limit per-frame scan logs
        try:
            for frame_tuple, meta in self.nextFrameEx():
                # GOP handling
                gop_id = meta.get('gop_id', 0)
                if across_gops:
                    if gop_id != 0 and DEBUG and _scan_count < 6:
                        print(f"[HEVC-DBG] gop change: gop_id={gop_id} at idx={meta.get('frame_index')} (ignored)")
                else:
                    if gop_id != 0:
                        if DEBUG and _scan_count < 6:
                            print(f"[HEVC-DBG] break at gop change: gop_id={gop_id} idx={meta.get('frame_index')}")
                        break

                if n_frames >= gop_size:
                    break

                is_i = bool(meta.get('is_i_frame', False))
                (
                    frame_type,
                    quadtree_stru,
                    rgb,
                    mv_x_L0,
                    mv_y_L0,
                    mv_x_L1,
                    mv_y_L1,
                    ref_off_L0,
                    ref_off_L1,
                    size,
                    residual,
                ) = frame_tuple

                if DEBUG and _scan_count < 6:
                    try:
                        _mvmax = float(max(np.abs(mv_x_L0).max(), np.abs(mv_y_L0).max()))
                        _refmax = int(np.max(ref_off_L0))
                    except Exception:
                        _mvmax, _refmax = -1.0, -1
                    _ft_id = meta.get('frame_type')
                    _ft_str = meta.get('frame_type_str')
                    print(f"[HEVC-DBG] scan idx={meta.get('frame_index')} g={gop_id} type={_ft_str}({_ft_id}) is_i={is_i} mvmax={_mvmax:.3f} refmaxL0={_refmax}")
                    _scan_count += 1

                def _collect_mv_res():
                    mvx, mvy = mv_x_L0, mv_y_L0
                    if upsample_mv:
                        mvx = self._upsample_mv_to_hw(mvx)
                        mvy = self._upsample_mv_to_hw(mvy)
                    if y_only:
                        if residual.ndim == 2:
                            res = residual[None, ...]
                        elif residual.ndim == 3:
                            res = residual[..., 0][None, ...]
                        else:
                            raise ValueError("Unexpected residual shape in Y-only mode: {}".format(residual.shape))
                    else:
                        if residual.ndim == 3:
                            res = cv2.cvtColor(residual, cv2.COLOR_BGR2RGB)
                            res = res.transpose(2,0,1)
                        else:
                            res = np.stack([residual]*3,axis=0)
                    p_mv_x.append(mvx)
                    p_mv_y.append(mvy)
                    p_res.append(res)
                    p_metas.append(meta)

                if is_i:
                    if collect_mv_from_all:
                        _collect_mv_res()
                    if need_i_rgb:
                        if y_only:
                            if rgb.ndim == 2:
                                i_frames.append(rgb[None, ...])
                            elif rgb.ndim == 3:
                                i_frames.append(rgb[..., 0][None, ...])
                            else:
                                raise ValueError("Unexpected rgb shape in Y-only mode: {}".format(rgb.shape))
                        else:
                            if rgb.ndim == 3:
                                rgb_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                                i_frames.append(rgb_img.transpose(2,0,1))
                            else:
                                i_frames.append(np.stack([rgb]*3,axis=0))
                        i_metas.append(meta)
                else:
                    _collect_mv_res()

                n_frames += 1
        finally:
            self.close()

        # Stack tensors
        tensors = {}
        if i_frames:
            i_arr = np.stack(i_frames, axis=0)  # [Ti, 1, H, W] or [Ti, 3, H, W]
            if y_only:
                # [Ti, 1, H, W]
                if i_arr.ndim == 3:
                    i_arr = i_arr[:,None,:,:]
                elif i_arr.shape[1] != 1:
                    i_arr = i_arr[:,None,:,:]
            else:
                if i_arr.ndim == 4 and i_arr.shape[1] == 3:
                    pass
                elif i_arr.ndim == 4 and i_arr.shape[-1] == 3:
                    i_arr = i_arr.transpose(0,3,1,2)
                elif i_arr.ndim == 3:
                    i_arr = np.stack([i_arr]*3,axis=1)
            tensors['I'] = i_arr
        else:
            tensors['I'] = np.zeros((0, n_i_col, H, W), dtype=np.uint8)
        if p_mv_x:
            mvx_arr = np.stack(p_mv_x, axis=0)  # [Tp, H, W]
            mvy_arr = np.stack(p_mv_y, axis=0)
            mv_arr = np.stack([mvx_arr, mvy_arr], axis=1)  # [Tp, 2, H, W]
            res_arr = np.stack(p_res, axis=0)  # [Tp, Cr, H, W]
            if res_arr.ndim == 4:
                pass
            elif res_arr.ndim == 3:
                res_arr = res_arr[:,None,:,:]
            tensors['MV'] = mv_arr
            tensors['R'] = res_arr
        else:
            tensors['MV'] = np.zeros((0, 2, H, W), dtype=np.float32)
            tensors['R'] = np.zeros((0, n_res_col, H, W), dtype=np.float32)

        # Normalize in-place
        if tensors['MV'].shape[0] > 0:
            mvx = tensors['MV'][:,0]
            mvy = tensors['MV'][:,1]
            res = tensors['R']
            mvx_norm, mvy_norm, res_norm = self._normalize_mv_and_res(
                mvx, mvy, res, mode=norm_mode, pct=mv_pct, signed_res=signed_res)
            tensors['MV'] = np.stack([mvx_norm, mvy_norm], axis=1)
            tensors['R'] = res_norm
        # I-frames: optional normalization (keep as uint8 or normalize if requested)
        # (No normalization for I-frames here; can add if needed)

        metas = i_metas + p_metas
        return tensors, metas



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for HevcFeatureReader")
    parser.add_argument("video", type=str, help="Path to an HEVC-encoded video (e.g., .mp4/.mkv)")
    parser.add_argument("--frames", type=int, default=2, help="Number of frames to read for sanity check")
    parser.add_argument("--out", type=str, default="hevc_debug", help="Output folder for dumps")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel factor for decoder")
    parser.add_argument("--y-only", type=int, choices=[0,1], default=None,
                        help="Override env UMT_HEVC_Y_ONLY for this run (1=Y-plane only, 0=decode BGR).")
    parser.add_argument("--viz-normalized", action="store_true",
                        help="Also visualize MV/Residual after model normalization (uses read_first_gop_tensors).")
    args = parser.parse_args()

    if args.y_only is not None:
        os.environ["UMT_HEVC_Y_ONLY"] = str(args.y_only)

    import json

    os.makedirs(args.out, exist_ok=True)
    rdr = HevcFeatureReader(args.video, nb_frames=args.frames, n_parallel=args.parallel)
    print(f"[hevc] video={args.video} size=({rdr.width}x{rdr.height}) frames={rdr.nb_frames} pix_fmt={rdr.pix_fmt}")

    # ---- Normalized visualization using read_first_gop_tensors (if requested) ----
    if args.viz_normalized:
        os.makedirs(args.out, exist_ok=True)
        # Inform the MV visualizer about full-res size (optional upsampling)
        viz_mv_to_hsv._full_H = rdr.height
        viz_mv_to_hsv._full_W = rdr.width

        # Use a separate reader because read_first_gop_tensors() closes its own process
        rdr_norm = HevcFeatureReader(args.video, nb_frames=args.frames, n_parallel=args.parallel)
        tensors, metas = rdr_norm.read_first_gop_tensors(
            gop_size=args.frames,
            need_i_rgb=True,
            mv_level='L0',
            upsample_mv=True,
            norm_mode='percentile',
            mv_pct=95,
            signed_res=True
        )
        rdr_norm.close()

        # --- MV (normalized): tensors['MV'] in [0,1]; convert to [-1,1] for direction-aware HSV ---
        if tensors['MV'].shape[0] > 0:
            Tp = tensors['MV'].shape[0]
            for t in range(min(args.frames-1, Tp)):
                mvx = tensors['MV'][t, 0] * 2.0 - 1.0
                mvy = tensors['MV'][t, 1] * 2.0 - 1.0
                mv_img = viz_mv_to_hsv(mvx.astype(np.float32), mvy.astype(np.float32))
                cv2.imwrite(os.path.join(args.out, f"{t:05d}_mv_L0_hsv.NORM.png"), mv_img)
                # Save arrays for inspection
                np.save(os.path.join(args.out, f"{t:05d}_mvx_norm.npy"), tensors['MV'][t, 0])
                np.save(os.path.join(args.out, f"{t:05d}_mvy_norm.npy"), tensors['MV'][t, 1])

        # --- Residual (normalized): tensors['R'] in [-1,1]; map to 0..255 for visualization and apply colormap ---
        if tensors['R'].shape[0] > 0:
            Rp = tensors['R'].shape[0]
            for t in range(min(args.frames-1, Rp)):
                r = tensors['R'][t]  # [C,H,W] or [H,W]
                if r.ndim == 3 and r.shape[0] == 1:
                    r_img = r[0]
                elif r.ndim == 3 and r.shape[0] == 3:
                    # If RGB residual: show magnitude of the first channel for a simple diverging map
                    r_img = r[0]
                else:
                    r_img = np.squeeze(r)
                # Clip to [-1,1], then map to 0..255 for visualization
                r_img = np.clip(r_img, -1.0, 1.0)
                r_vis = ((r_img + 1.0) * 127.5).astype(np.uint8)
                r_vis = cv2.applyColorMap(r_vis, cv2.COLORMAP_TURBO)
                cv2.imwrite(os.path.join(args.out, f"{t:05d}_residual_viz.NORM.png"), r_vis)
                # Save array for inspection
                np.save(os.path.join(args.out, f"{t:05d}_residual_norm.npy"), r)


    for i, item in enumerate(rdr.nextFrameEx()):
        if i >= args.frames:
            break
        (frame_tuple, meta) = item
        (
            frame_type,
            quadtree_stru,
            rgb,
            mv_x_L0,
            mv_y_L0,
            mv_x_L1,
            mv_y_L1,
            ref_off_L0,
            ref_off_L1,
            size,
            residual,
        ) = frame_tuple

        print(f"[hevc] frame#{i} type={meta['frame_type_str']}({int(frame_type)}) gop={meta['gop_id']} ts={meta['timestamp']} rgb={rgb.shape} residual={residual.shape} mvL0=({mv_x_L0.shape},{mv_y_L0.shape}) size={size.shape}")
        mvmaxL0 = max(np.abs(mv_x_L0).max(), np.abs(mv_y_L0).max())
        refmaxL0 = int(np.max(ref_off_L0))
        print(f"[hevc] mvmaxL0={mvmaxL0} refmaxL0={refmaxL0}")

        # Share full-res H,W for MV visualizer to optionally upsample quarter-res inputs
        viz_mv_to_hsv._full_H = rdr.height
        viz_mv_to_hsv._full_W = rdr.width

        # Visualize MV (L0) as color wheel (BGR)
        mv_bgr = viz_mv_to_hsv(mv_x_L0.astype(np.float32), mv_y_L0.astype(np.float32))
        cv2.imwrite(os.path.join(args.out, f"{i:05d}_mv_L0_hsv.png"), mv_bgr)

        # Visualize residual heatmap (assume signed around 128 for Y)
        res_vis = viz_residual(residual, signed=True)
        cv2.imwrite(os.path.join(args.out, f"{i:05d}_residual_viz.png"), res_vis)

        # Save RGB and residual with proper channel handling
        rgb_to_save = rgb
        if isinstance(rgb_to_save, np.ndarray) and rgb_to_save.ndim == 2:
            # Y-only mode -> expand to 3-channel BGR for visualization
            rgb_to_save = cv2.cvtColor(rgb_to_save, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(args.out, f"{i:05d}_rgb.png"), rgb_to_save)

        residual_to_save = residual
        # Residual can be Y or BGR; keep as-is for raw dump
        cv2.imwrite(os.path.join(args.out, f"{i:05d}_residual.png"), residual_to_save)

        np.save(os.path.join(args.out, f"{i:05d}_mvx_L0.npy"), mv_x_L0)
        np.save(os.path.join(args.out, f"{i:05d}_mvy_L0.npy"), mv_y_L0)
        np.save(os.path.join(args.out, f"{i:05d}_ref_off_L0.npy"), ref_off_L0)
        np.save(os.path.join(args.out, f"{i:05d}_size.npy"), size)

        with open(os.path.join(args.out, f"{i:05d}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    rdr.close()
    print(f"[hevc] dumps written to: {args.out}")
    