import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from sklearn.preprocessing import normalize

def process_npy_file(args):
    """处理单个NPY文件，输入数据形状应为 [num_features, num_frames, dims]"""
    file_path, dims, frames_out, output_dir = args
    try:
        data = np.load(file_path)
        if not isinstance(data, np.ndarray):
            print(f"文件 {file_path} 加载后不是numpy数组，跳过")
            return file_path, False

        if data.ndim != 3:
            print(f"文件 {file_path} 的shape为{data.shape}，不是3维 [num_features, num_frames, dims]，跳过")
            return file_path, False

        num_features, num_frames, D = data.shape

        # 选择将要使用的维度数
        if dims is None:
            use_dims = D
        else:
            if dims <= 0:
                print(f"文件 {file_path} 的 --dims={dims} 非法，必须为正数，跳过")
                return file_path, False
            if dims > D:
                print(f"文件 {file_path} 的 dims({dims}) 大于特征维度({D})，将自动截断为 {D}")
                use_dims = D
            else:
                use_dims = dims

        # 提取前 use_dims 维
        processed = data[:, :, :use_dims]  # [N, T, use_dims]

        # 逐帧归一化（对最后一维特征向量做L2归一化）
        flat = processed.reshape(-1, use_dims)  # [(N*T), use_dims]
        flat = normalize(flat, axis=1)
        processed = flat.reshape(num_features, num_frames, use_dims)

        # 校验 frames_out（目标输出帧数）
        if not isinstance(frames_out, int) or frames_out <= 0:
            print(f"文件 {file_path} 的 frames_out={frames_out} 非法，必须为正整数，跳过")
            return file_path, False
        if frames_out > num_frames:
            print(f"文件 {file_path} 的 frames_out({frames_out}) 必须小于 num_frames({num_frames})，跳过")
            return file_path, False

        # 要求能整除，否则报错
        if num_frames % frames_out != 0:
            print(f"文件 {file_path}: num_frames({num_frames}) 不能被 frames_out({frames_out}) 整除，跳过")
            return file_path, False

        # 计算步长并按步长均匀抽帧（等价于切片 [::step]），确保输出帧数为 frames_out
        step = num_frames // frames_out
        processed = processed[:, ::step, :]  # [N, frames_out, use_dims]
        if processed.shape[1] != frames_out:
            print(f"文件 {file_path}: 抽帧后帧数({processed.shape[1]}) != 期望({frames_out})，跳过")
            return file_path, False

        # 输出二维数组：(N, frames_out * use_dims)
        out2d = processed.reshape(-1, frames_out * use_dims)
        # 再次归一化（可选）
        out2d = normalize(out2d, axis=1)

        # 保存到输出目录
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(file_path)
        if base_filename.lower().endswith(".npy"):
            base_filename = base_filename[:-4]
        output_path = os.path.join(output_dir, f"processed_{base_filename}_frames{frames_out}.npy")

        np.save(output_path, out2d)
        return file_path, True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return file_path, False

def collect_npy_files_from_dir(input_dir, recursive=False):
    files = []
    if recursive:
        for root, _, filenames in os.walk(input_dir):
            for fn in filenames:
                if fn.lower().endswith(".npy"):
                    files.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_dir):
            fp = os.path.join(input_dir, fn)
            if os.path.isfile(fp) and fn.lower().endswith(".npy"):
                files.append(fp)
    return files

def collect_npy_files(input_path, recursive=False):
    """input_path 可以是目录或列表文件"""
    if os.path.isdir(input_path):
        return collect_npy_files_from_dir(input_path, recursive=recursive)
    if os.path.isfile(input_path):
        # 将其视为列表文件
        files = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if not p.lower().endswith(".npy"):
                    continue
                if not os.path.isabs(p):
                    # 相对路径以列表文件所在目录为基准
                    p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(input_path)), p))
                if os.path.isfile(p):
                    files.append(p)
                else:
                    print(f"列表中路径不存在，已忽略: {p}")
        return files
    print(f"输入路径既不是目录也不是文件: {input_path}")
    return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="处理NPY特征文件（形状 [num_features, num_frames, dims]）；支持目录或列表文件输入；按目标输出帧数均匀抽帧")
    parser.add_argument("--input", type=str, required=True, help="输入源：目录（含NPY）或包含NPY文件路径的txt列表")
    parser.add_argument("--recursive", action="store_true", help="若 --input 为目录，是否递归查找子目录中的NPY文件")
    parser.add_argument("--num_processes", type=int, default=64, help="使用的进程数（最大不超过64，默认CPU核心数与64两者的较小值）")
    parser.add_argument("--dims", type=int, default=512, help="要提取的特征维度数（沿最后一维截取前dims；默认保留全部）")
    parser.add_argument("--frames_out", type=int, default=8, help="目标输出帧数，要求 0 < frames_out < num_frames 且 num_frames 能被 frames_out 整除")
    parser.add_argument("--output_suffix", type=str, default=None, help="输出目录后缀（默认：_processed_dim_{dims or 'all'}_frames_{frames_out}）")
    args = parser.parse_args()

    # 收集文件
    npy_files = collect_npy_files(args.input, recursive=args.recursive)
    if not npy_files:
        print("未找到任何NPY文件")
        return
    npy_files = sorted(npy_files)
    print("开始处理NPY文件...")
    print(f"待处理 NPY 文件数: {len(npy_files)}")
    print(f"dims={'全部' if args.dims is None else args.dims}，frames_out={args.frames_out}")

    # 输出目录（为输入目录或列表文件所在目录追加后缀）
    if args.output_suffix is None:
        suffix = f"_processed_dim_{args.dims if args.dims is not None else 'all'}_frames_{args.frames_out}"
    else:
        suffix = args.output_suffix

    if os.path.isdir(args.input):
        base_dir = args.input.rstrip('/\\')
        output_dir = base_dir + suffix
    else:
        # 列表文件的父目录 + 后缀
        parent_dir = os.path.dirname(os.path.abspath(args.input)).rstrip('/\\')
        output_dir = parent_dir + suffix

    print(f"输出目录: {output_dir}")

    # 进程数限制最大32
    cpu_cnt = mp.cpu_count()
    if args.num_processes is None or args.num_processes <= 0:
        num_processes = min(cpu_cnt, 64)
    else:
        num_processes = min(args.num_processes, 64)
    print(f"使用 {num_processes} 个进程并行处理（CPU:{cpu_cnt}，上限:64")

    # 参数打包
    process_args = [(file_path, args.dims, args.frames_out, output_dir) for file_path in npy_files]

    # 并行处理
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_npy_file, process_args),
            total=len(npy_files),
            desc="处理NPY文件"
        ))

    # 统计
    success_files = [fp for fp, ok in results if ok]
    print(f"成功处理了 {len(success_files)}/{len(npy_files)} 个文件")
    print("处理完成！")

if __name__ == "__main__":
    main()