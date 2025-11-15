import os
import subprocess as sp
import struct
import numpy as np
from typing import Generator, Optional, Tuple, Union

# Binary header layout (must match C): '<IHHHHBBI'
# magic='RES1'(0x31534552), width, height, pitchY, pitchC, channels(bit0:Y,bit1:U,bit2:V), reserved, data_len
_RES_MAGIC = 0x31534552
_HDR_FMT = '<IHHHHBBI'
_HDR_SIZE = struct.calcsize(_HDR_FMT)

_HEVC_FEAT_DECODER = os.environ.get('HEVC_FEAT_DECODER', '/video_vit/yunyaoyan/openHEVC_feature_decoder-Interface_MV_Residual/openHEVC_feature_decoder-Interface_MV_Residual/build/hevc')

class ResPipeReader:
    """
    Reader for the binary framed residual-only pipe produced by the HEVC feature decoder
    when compiled with USE_LEGACY_FWRITE=0 ("length-header + payload" per frame).

    Each frame:
      Header:  struct '<IHHHHBBI'
      Payload: Y residual (always), then optionally U residual, then V residual
               depending on header.channels bits. All planes are uint8 and laid
               out row-major with their respective pitch (stride).

    Notes:
      - We assume yuv420 for chroma sizes (height//2). This matches the encoder
        and the C writer.
      - Returned arrays are cropped to the logical width (header.width). Pitch
        may be >= width due to alignment, but we slice so the caller sees (H,W)
        and (H//2, W//2).
    """

    def __init__(self, filename: str, nb_frames: Optional[int] = None, n_parallel: int = 1,
                 decoder_path: Optional[str] = None) -> None:
        self.filename = filename
        self.nb_frames = nb_frames
        self.n_parallel = int(n_parallel)
        self.decoder_path = decoder_path or _HEVC_FEAT_DECODER

        if not os.path.isfile(self.decoder_path) or not os.access(self.decoder_path, os.X_OK):
            raise FileNotFoundError(
                f"HEVC feature decoder not found or not executable at '{self.decoder_path}'.\n"
                f"Set env HEVC_FEAT_DECODER to the correct binary path.")
        if not os.path.exists(self.filename):
            raise FileNotFoundError(self.filename)

        # Launch decoder (binary protocol on stdout). Keep stderr for debugging.
        self._proc = sp.Popen(
            [self.decoder_path, '-i', self.filename, '-p', str(self.n_parallel)],
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
        )
        self._stdout = self._proc.stdout
        self._closed = False
        # will be populated from first header
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.pitchY: Optional[int] = None
        self.pitchC: Optional[int] = None
        self.channels: Optional[int] = None

    def _read_exact(self, n: int) -> bytes:
        assert self._stdout is not None
        buf = self._stdout.read(n)
        if buf is None or len(buf) != n:
            # Propagate EOF as None
            return b''
        return buf

    def _read_one(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        # Read header
        hdr = self._read_exact(_HDR_SIZE)
        if not hdr:
            return None  # EOF
        magic, width, height, pitchY, pitchC, channels, _reserved, data_len = struct.unpack(_HDR_FMT, hdr)
        if magic != _RES_MAGIC:
            raise RuntimeError(f"Bad magic in residual stream: got 0x{magic:08x}, expected 0x{_RES_MAGIC:08x}")

        self.width, self.height = int(width), int(height)
        self.pitchY, self.pitchC = int(pitchY), int(pitchC)
        self.channels = int(channels)

        payload = self._read_exact(int(data_len))
        if not payload or len(payload) != int(data_len):
            raise RuntimeError(f"Short read in payload: expected {data_len}, got {0 if not payload else len(payload)}")

        # Slice planes from payload
        off = 0
        szY = self.pitchY * self.height
        y = np.frombuffer(payload, dtype=np.uint8, count=szY, offset=off).reshape(self.height, self.pitchY)[:, :self.width]
        off += szY

        u = v = None
        # We assume yuv420 if chroma present
        if self.channels & 0x02:  # U present
            h2 = self.height >> 1
            w2 = self.width >> 1
            szU = self.pitchC * h2
            u = np.frombuffer(payload, dtype=np.uint8, count=szU, offset=off).reshape(h2, self.pitchC)[:, :w2]
            off += szU
        if self.channels & 0x04:  # V present
            h2 = self.height >> 1
            w2 = self.width >> 1
            szV = self.pitchC * h2
            v = np.frombuffer(payload, dtype=np.uint8, count=szV, offset=off).reshape(h2, self.pitchC)[:, :w2]
            off += szV

        return y, u, v

    def next_residual(self) -> Generator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], None, None]:
        """Yield residual per frame. If only Y is present, yields a single (H,W) array.
        Otherwise yields a tuple (Y, U, V). Stops at nb_frames if provided, else on EOF."""
        count = 0
        while True:
            if self.nb_frames is not None and count >= int(self.nb_frames):
                return
            frame = self._read_one()
            if frame is None:
                return
            y, u, v = frame
            count += 1
            if (u is None) and (v is None):
                yield y
            else:
                # If one chroma plane missing, synthesize None to keep arity consistent
                yield (y, u, v)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._proc is not None:
                try:
                    if self._proc.stdin:
                        self._proc.stdin.close()
                except Exception:
                    pass
                try:
                    if self._proc.stdout:
                        self._proc.stdout.close()
                except Exception:
                    pass
                try:
                    if self._proc.stderr:
                        self._proc.stderr.close()
                except Exception:
                    pass
                if self._proc.poll() is None:
                    self._proc.terminate()
        finally:
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Read residuals from binary framed HEVC residual pipe')
    parser.add_argument('video', type=str, help='Path to input video (will be passed to decoder)')
    parser.add_argument('--frames', type=int, default=None, help='Max frames to read (default: until EOF)')
    parser.add_argument('--parallel', type=int, default=8, help='Decoder parallel factor (-p)')
    parser.add_argument('--decoder', type=str, default=None, help='Override HEVC feature decoder path')
    args = parser.parse_args()

    rdr = ResPipeReader(args.video, nb_frames=args.frames, n_parallel=args.parallel, decoder_path=args.decoder)
    print(f"[respipe] open video={args.video}")
    n = 0
    try:
        for fr in rdr.next_residual():
            if isinstance(fr, tuple):
                y,u,v = fr
                print(f"[respipe] frame#{n}: Y={y.shape} U={None if u is None else u.shape} V={None if v is None else v.shape}")
            else:
                print(f"[respipe] frame#{n}: Y={fr.shape}")
            n += 1
            if args.frames is not None and n >= args.frames:
                break
    finally:
        rdr.close()
    print(f"[respipe] done, frames={n}")
