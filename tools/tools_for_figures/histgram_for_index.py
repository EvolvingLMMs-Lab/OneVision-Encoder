import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

plt.rcParams.update({
    "figure.dpi": 250,
    "savefig.dpi": 1200,
    "axes.titlesize": 27,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    # "axes.facecolor": "#FFFFFF",
    # "figure.facecolor": "#FFFFFF",
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})

def load_flat_indices(path: Path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0:
        try:
            obj = arr.item()
            if isinstance(obj, dict):
                for k in ["indices", "idx", "token_indices", "tokens", "flat_indices", "selected"]:
                    if k in obj:
                        return np.asarray(obj[k]).ravel()
                return np.asarray(obj).ravel()
            else:
                return np.asarray(obj).ravel()
        except Exception:
            return np.asarray(arr).ravel()
    else:
        return np.asarray(arr).ravel()

def flat_to_thw(flat: np.ndarray, T: int):
    H, W = 14, 14
    hw = H * W
    t = flat // hw
    remain = flat % hw
    h = remain // W
    w = remain % W
    return t, h, w

def compute_spatial_prob(codec_dir: Path, T: int, topk: int, max_files: int, min_t: int):
    H, W = 14, 14
    files = sorted(codec_dir.glob("*.npy"))
    if max_files > 0:
        files = files[:max_files]
    spatial_counts = np.zeros((H, W), dtype=np.float64)
    for f in tqdm(files, desc="Codec aggregation"):
        flat = load_flat_indices(f)
        if topk > 0:
            flat = flat[:topk]
        mask = (flat >= 0) & (flat < T * H * W)
        flat = flat[mask]
        if flat.size == 0:
            continue
        t, h, w = flat_to_thw(flat, T)
        if min_t > 0:
            keep = t >= min_t
            h = h[keep]; w = w[keep]
        np.add.at(spatial_counts, (h, w), 1)
    spatial_prob = spatial_counts / spatial_counts.sum()
    return spatial_prob

def plot_surface_with_grid(spatial_prob: np.ndarray, out_png: Path, out_pdf: Path,
                          sigma: float = 0.55,
                          cmap_name: str = "coolwarm",
                          azim: float = -55,
                          elev: float = 24,
                          z_label: str = r"Token Probability",
                          colorbar_loc: str = "bottom",
                          value_fmt: str = ".4f",
                          show_contour: bool = False,
                          transparent: bool = False):
    H, W = spatial_prob.shape
    axis_ticks = np.arange(0, 14, 1)
    X, Y = np.meshgrid(np.arange(H), np.arange(W))
    Z = gaussian_filter(spatial_prob, sigma=sigma, mode="nearest")
    norm = plt.Normalize(np.percentile(Z, 0.5), np.percentile(Z, 99.5))
    cmap = cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(10, 10),)
    ax = fig.add_subplot(111, projection="3d",)
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, edgecolor='#DDDDDD',
                          linewidth=1.1, antialiased=True, alpha=0.90, rstride=1, cstride=1)
    ax.plot_wireframe(X, Y, Z, color='#B7B7B7', linewidth=0.35, alpha=0.6, rstride=1, cstride=1)

    ax.grid(True, color="#BBBBBB", linestyle="dotted", linewidth=1.1)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    zmin, zmax = float(np.min(Z)), float(np.max(Z))
    zticks = np.linspace(zmin, zmax, 7)
    ax.set_zticks(zticks)
    def z_perc_fmt(v, pos):
        return f"{v*100:.2f}%"
    ax.zaxis.set_major_formatter(plt.FuncFormatter(z_perc_fmt))

    ax.tick_params(axis='x', which='major', pad=5)
    ax.tick_params(axis='y', which='major', pad=5)
    ax.tick_params(axis='z', which='major', pad=20)
    ax.set_xlabel(r"$H$", fontsize=19, labelpad=16, fontweight="semibold")
    ax.set_ylabel(r"$W$", fontsize=19, labelpad=16, fontweight="semibold")
    ax.set_box_aspect([1.13, 1.13, 0.85])
    ax.view_init(elev=elev, azim=azim)

    if show_contour:
        levels = np.linspace(zmin, zmax, 13)
        ax.contour(X, Y, Z, zdir='z', offset=zmin, levels=levels, cmap=cmap,
                   alpha=0.3, linestyles="dashed")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z)
    def cb_perc_fmt(v, pos): return f"{v*100:.2f}%"
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        pad=0.04,
        fraction=0.07,
        shrink=0.9,
        # aspect=28,
        location=colorbar_loc
    )
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(cb_perc_fmt))
    cbar.ax.tick_params(labelsize=13)

    # fig.patch.set_facecolor('white')
    fig.tight_layout(pad=3.0)
    fig.savefig(out_png, bbox_inches="tight", transparent=transparent)
    fig.savefig(out_pdf, bbox_inches="tight", transparent=False)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="3D surface with grid ticks for Codec spatial probability")
    ap.add_argument("--codec-dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--max-files", type=int, default=1000)
    ap.add_argument("--min-t", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.55)
    ap.add_argument("--azim", type=float, default=-55)
    ap.add_argument("--elev", type=float, default=24)
    ap.add_argument("--zlabel", type=str, default="Token Probability")
    ap.add_argument("--minmax-format", type=str, default=".4f")
    ap.add_argument("--show-contour", action="store_true", default=False)
    ap.add_argument("--out-dir", type=str, default="./codec_surface_whitegrid")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    spatial_prob = compute_spatial_prob(
        Path(args.codec_dir), args.T, args.topk, args.max_files, args.min_t)

    themes = [
        "coolwarm",
        # "viridis", "plasma", "inferno",
        # "magma", "cividis", "YlGnBu", "RdYlBu",
        # "Spectral", "rainbow"
    ]
    for cmap in themes:
        out_png = out_dir / f"codec_spatial_surface_whitegrid__{cmap}.png"
        out_pdf = out_dir / f"codec_spatial_surface_whitegrid__{cmap}.pdf"
        plot_surface_with_grid(spatial_prob, out_png, out_pdf,
            sigma=args.sigma,
            cmap_name=cmap,
            azim=args.azim,
            elev=args.elev,
            z_label=args.zlabel,
            value_fmt=args.minmax_format,
            show_contour=args.show_contour,
            transparent=True  # <--- 关键！PNG为透明
        )
        print(f"[DONE] Theme: {cmap} | PNG: {out_png} | PDF: {out_pdf}")

    print('Use --show-contour to add projected contour lines.')

if __name__ == "__main__":
    main()
