#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import multiprocessing
import os
import pickle
import random
import tempfile
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import streamlit as st


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频聚类样本可视化工具")
    parser.add_argument("--index_file", type=str, required=True, help="聚类结果文件路径 (如 inverted_index.pkl)")
    parser.add_argument("--video_list", type=str, required=True, help="视频文件列表路径")
    return parser.parse_args()

# 设置页面配置
st.set_page_config(
    page_title="视频聚类样本可视化工具",
    layout="wide",
    menu_items={
        'About': "# 视频聚类样本可视化工具\n使用Streamlit构建的视频聚类数据可视化应用"
    }
)

# 使用缓存机制缓存文件列表
@st.cache_resource
def load_video_list(list_path: str) -> List[str]:
    """
    加载视频文件列表

    Args:
        list_path: 视频列表文件路径

    Returns:
        视频文件路径列表
    """
    try:
        print(f"加载视频列表: {list_path}")
        video_paths = []

        with open(list_path, 'r') as f:
            for line in f:
                video_path = line.strip()
                if video_path:  # 忽略空行
                    video_paths.append(video_path)

        print(f"成功加载视频列表，包含 {len(video_paths)} 个视频")
        return video_paths
    except Exception as e:
        st.error(f"加载视频列表时出错: {str(e)}")
        return []

# 使用缓存机制缓存倒排索引
@st.cache_resource
def load_inverted_index(index_path: str) -> Dict:
    """
    加载倒排索引（只存储索引而非实际路径，提高性能）

    Args:
        index_path: 索引文件路径

    Returns:
        倒排索引字典，每个簇对应一组视频索引
    """
    try:
        print(f"加载倒排索引: {index_path}")
        if index_path.endswith('.pkl'):
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                print(f"成功加载pickle文件，包含 {len(data)} 个簇")
                return data
        elif index_path.endswith('.json'):
            import json
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"成功加载JSON文件，包含 {len(data)} 个簇")
                return data
        else:
            st.error(f"不支持的索引文件格式: {index_path}")
            return {}
    except Exception as e:
        import traceback
        st.error(f"加载倒排索引时出错: {str(e)}")
        st.code(traceback.format_exc())
        return {}

def get_paths_for_cluster(cluster_indices: List[int], video_list: List[str]) -> List[str]:
    """
    只为当前选定的簇转换索引到路径

    Args:
        cluster_indices: 簇中视频的索引列表
        video_list: 所有视频的路径列表

    Returns:
        簇中视频的路径列表
    """
    valid_paths = []
    for idx in cluster_indices:
        if 0 <= idx < len(video_list):
            valid_paths.append(video_list[idx])
    return valid_paths

@st.cache_data
def extract_frame_from_video(video_path: str, frame_index: int = 0) -> Optional[bytes]:
    """
    从视频中提取特定帧作为缩略图（使用st.cache_data提高性能）

    Args:
        video_path: 视频文件路径
        frame_index: 要提取的帧索引

    Returns:
        帧图像的二进制数据
    """
    try:
        if not os.path.exists(video_path):
            return None

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查是否成功打开
        if not cap.isOpened():
            return None

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return None

        # 调整帧索引，确保有效
        frame_index = min(frame_index, total_frames - 1)

        # 设置视频位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取帧
        ret, frame = cap.read()

        # 释放视频
        cap.release()

        if not ret or frame is None:
            return None

        # 转换为JPEG格式
        _, buffer = cv2.imencode('.jpg', frame)

        return buffer.tobytes()

    except Exception as e:
        print(f"提取视频帧时出错: {str(e)}")
        return None

# 不使用缓存的GIF转换函数，用于多进程调用
def _convert_video_to_gif_worker(video_path: str, max_frames: int = 50, fps: int = 10,
                                resize_factor: float = 0.5) -> Optional[Tuple[str, str]]:
    """
    将视频转换为GIF格式的工作函数（用于多进程）

    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，限制GIF大小
        fps: 每秒帧数
        resize_factor: 调整大小的因子（0.5表示原尺寸的一半）

    Returns:
        (视频路径, Base64编码的GIF数据) 元组
    """
    try:
        if not os.path.exists(video_path):
            return (video_path, None)

        # 创建临时文件保存GIF
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
            gif_path = tmp_file.name

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查是否成功打开
        if not cap.isOpened():
            return (video_path, None)

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算新尺寸
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)

        # 计算采样间隔，确保不超过max_frames
        sampling_interval = max(1, total_frames // max_frames)

        # 收集帧
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只收集每sampling_interval帧
            if frame_count % sampling_interval == 0:
                # 调整帧大小
                resized_frame = cv2.resize(frame, (new_width, new_height))
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

            frame_count += 1

            # 限制最大帧数
            if len(frames) >= max_frames:
                break

        # 释放视频
        cap.release()

        if not frames:
            return (video_path, None)

        # 生成GIF
        imageio.mimsave(gif_path, frames, fps=fps, format='GIF')

        # 读取GIF文件并转换为base64
        with open(gif_path, 'rb') as f:
            gif_data = f.read()

        # 删除临时文件
        os.unlink(gif_path)

        # 转换为base64
        b64_gif = base64.b64encode(gif_data).decode('utf-8')

        # 返回data URI
        return (video_path, f"data:image/gif;base64,{b64_gif}")

    except Exception as e:
        import traceback
        print(f"转换视频到GIF时出错 ({video_path}): {str(e)}")
        print(traceback.format_exc())
        return (video_path, None)

# 使用多进程批量转换视频为GIF
def batch_convert_videos_to_gifs(video_paths: List[str], max_frames: int = 50,
                               fps: int = 10, resize_factor: float = 0.5) -> Dict[str, str]:
    """
    使用多进程并行将多个视频转换为GIF

    Args:
        video_paths: 视频路径列表
        max_frames: 最大帧数
        fps: 帧率
        resize_factor: 缩放比例

    Returns:
        字典，键为视频路径，值为对应的GIF数据URI
    """
    # 检查视频路径列表是否为空
    if not video_paths:
        return {}

    # 创建一个进程池，进程数等于CPU核心数或视频数量（取较小值）
    num_processes = min(multiprocessing.cpu_count(), len(video_paths))

    # 显示进度
    start_time = time.time()
    st.write(f"正在使用 {num_processes} 个进程并行处理 {len(video_paths)} 个视频...")

    # 准备转换函数
    convert_func = partial(
        _convert_video_to_gif_worker,
        max_frames=max_frames,
        fps=fps,
        resize_factor=resize_factor
    )

    results = {}

    try:
        # 使用进程池并行处理视频
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 启动异步任务
            result_objects = [
                pool.apply_async(convert_func, (video_path,))
                for video_path in video_paths
            ]

            # 准备进度条
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # 收集结果
            for i, result_obj in enumerate(result_objects):
                # 更新进度条
                progress = (i + 1) / len(result_objects)
                progress_bar.progress(progress)
                progress_text.text(f"处理进度: {i+1}/{len(result_objects)} ({progress*100:.1f}%)")

                # 获取结果
                try:
                    video_path, gif_data = result_obj.get(timeout=30)  # 设置超时时间
                    if gif_data:
                        results[video_path] = gif_data
                except Exception as e:
                    print(f"获取GIF转换结果时出错: {str(e)}")

        # 清理进度显示
        progress_bar.empty()
        progress_text.empty()

        # 显示处理时间
        end_time = time.time()
        st.write(f"GIF处理完成，用时 {end_time - start_time:.2f} 秒")

    except Exception as e:
        import traceback
        print(f"多进程转换视频失败: {str(e)}")
        print(traceback.format_exc())

    return results

# 获取视频基本信息
@st.cache_data
def get_video_info(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    """
    获取视频的基本信息

    Args:
        video_path: 视频文件路径

    Returns:
        (总帧数, 帧率, (宽度, 高度))元组
    """
    try:
        if not os.path.exists(video_path):
            return (0, 0.0, (0, 0))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (0, 0.0, (0, 0))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        return (total_frames, fps, (width, height))
    except Exception as e:
        print(f"获取视频信息时出错: {str(e)}")
        return (0, 0.0, (0, 0))

# 批量获取多个视频的信息
def batch_get_video_info(video_paths: List[str]) -> Dict[str, Tuple[int, float, Tuple[int, int]]]:
    """
    获取多个视频的基本信息

    Args:
        video_paths: 视频路径列表

    Returns:
        字典，键为视频路径，值为视频信息元组
    """
    results = {}
    for path in video_paths:
        results[path] = get_video_info(path)
    return results

# 优化：获取簇的路径列表而不加载实际视频
def get_cluster_paths_only(clusters, selected_cluster_idx):
    """
    获取所选簇的视频路径

    Args:
        clusters: 所有簇的列表
        selected_cluster_idx: 选中的簇索引

    Returns:
        选中簇的标签和视频路径列表
    """
    selected_label, selected_files = clusters[selected_cluster_idx]
    return selected_label, selected_files

# 优化：只获取当前页的视频路径
def get_current_page_paths(selected_files, page_number, samples_per_page):
    """
    获取当前页的视频路径

    Args:
        selected_files: 所选簇的所有视频路径
        page_number: 当前页码
        samples_per_page: 每页显示的样本数

    Returns:
        当前页的视频路径、起始索引和结束索引
    """
    start_idx = (page_number - 1) * samples_per_page
    end_idx = min(start_idx + samples_per_page, len(selected_files))
    current_page_files = selected_files[start_idx:end_idx]
    return current_page_files, start_idx, end_idx

def render_gif_html(gif_data_uri: str, width: str = "100%") -> None:
    """
    渲染GIF的HTML代码

    Args:
        gif_data_uri: 包含GIF数据的data URI
        width: GIF宽度

    Returns:
        None
    """
    html_code = f"""
    <img src="{gif_data_uri}" width="{width}" style="display:block; margin:auto;">
    """
    st.markdown(html_code, unsafe_allow_html=True)

# Streamlit应用
def main():
    # 获取命令行参数
    try:
        args = parse_arguments()
        index_file = args.index_file
        video_list_file = args.video_list
    except SystemExit:
        # 当在Streamlit中运行时，无法解析命令行参数，使用下面的默认值
        # 这些值可以通过Streamlit的命令行选项覆盖
        if 'index_file' in st.session_state and 'video_list_file' in st.session_state:
            index_file = st.session_state.index_file
            video_list_file = st.session_state.video_list_file
        else:
            # 首次运行时允许用户输入
            st.title("视频聚类样本可视化工具")
            st.write("请输入必要的文件路径：")

            # 获取文件路径
            index_file = st.text_input("聚类结果文件路径 (inverted_index.pkl):", "/path/to/inverted_index.pkl")
            video_list_file = st.text_input("视频列表文件路径:", "/path/to/video_list.txt")

            if st.button("确认"):
                if index_file and video_list_file:
                    # 保存路径到会话状态
                    st.session_state.index_file = index_file
                    st.session_state.video_list_file = video_list_file
                    st.rerun()
                else:
                    st.error("请输入所有必要的文件路径")

            # 首次运行时只显示输入表单
            return

    st.title("视频聚类样本可视化工具")

    # 侧边栏配置
    st.sidebar.header("配置")

    # 显示当前使用的文件
    st.sidebar.subheader("当前文件")
    st.sidebar.text(f"聚类文件: {os.path.basename(index_file)}")
    st.sidebar.text(f"视频列表: {os.path.basename(video_list_file)}")

    # 修改文件路径
    if st.sidebar.button("修改文件路径"):
        # 清除会话状态中的文件路径
        if 'index_file' in st.session_state:
            del st.session_state.index_file
        if 'video_list_file' in st.session_state:
            del st.session_state.video_list_file
        st.rerun()

    # 配置参数
    st.sidebar.subheader("可视化配置")

    # 每页显示的样本数
    samples_per_page = st.sidebar.number_input("每页样本数", min_value=1, max_value=20, value=8)

    # 每行显示的样本数
    samples_per_row = st.sidebar.number_input("每行样本数", min_value=1, max_value=4, value=2)

    # GIF相关配置
    st.sidebar.subheader("GIF配置")

    # GIF帧率
    gif_fps = st.sidebar.slider("GIF帧率", min_value=1, max_value=30, value=10)

    # GIF最大帧数
    gif_max_frames = st.sidebar.slider("GIF最大帧数", min_value=10, max_value=100, value=50)

    # GIF尺寸缩放因子
    gif_resize_factor = st.sidebar.slider("GIF缩放比例", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    # 多进程配置
    st.sidebar.subheader("多进程配置")
    use_multiprocessing = st.sidebar.checkbox("使用多进程加速GIF生成", value=True)

    # 处理倒排索引
    try:
        # 显示所选文件路径
        st.write(f"聚类文件: **{os.path.basename(index_file)}** ({index_file})")
        st.write(f"视频列表: **{os.path.basename(video_list_file)}** ({video_list_file})")

        # 检查文件是否存在
        if not os.path.exists(index_file):
            st.error(f"聚类文件不存在: {index_file}")
            return

        if not os.path.exists(video_list_file):
            st.error(f"视频列表文件不存在: {video_list_file}")
            return

        # 加载视频列表
        with st.spinner("加载视频列表..."):
            video_list = load_video_list(video_list_file)
            if not video_list:
                st.error("视频列表为空或无法加载")
                return

        # 加载倒排索引 (只读取一次，将被缓存)
        with st.spinner("加载聚类数据..."):
            inverted_index = load_inverted_index(index_file)

            if not inverted_index:
                st.error("聚类文件为空或格式错误")
                return

        # 按簇大小排序
        clusters = [(label, indices) for label, indices in inverted_index.items()]
        clusters.sort(key=lambda x: len(x[1]), reverse=True)

        # 显示簇信息
        st.header("按大小排序的簇")

        # 初始化随机idx（如果还未设置）
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = random.randint(0, len(clusters) - 1)

        # 显示当前簇的信息作为提示
        st.write(f"总共有 {len(clusters)} 个簇，范围为 0-{len(clusters)-1}")

        # 使用number_input代替selectbox，并设置初始值为随机数
        selected_cluster_idx = st.number_input(
            "输入要查看的簇索引",
            min_value=0,
            max_value=len(clusters)-1,
            value=st.session_state.random_idx,
            step=1
        )

        # 获取所选簇的路径（不加载视频）
        selected_label, selected_indices = clusters[selected_cluster_idx]
        selected_files = get_paths_for_cluster(selected_indices, video_list)
        st.write(f"选择的簇: {selected_label}, 包含 {len(selected_files)} 个样本")

        # 随机排序样本列表
        if 'sorted_files_key' not in st.session_state or st.session_state.sorted_files_key != selected_cluster_idx:
            random.seed(42)  # 使用固定种子以确保每次排序结果一致
            random.shuffle(selected_files)
            st.session_state.sorted_files_key = selected_cluster_idx

        # 分页控制
        total_pages = max(1, (len(selected_files) + samples_per_page - 1) // samples_per_page)
        page_number = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

        # 获取当前页的视频路径（不加载视频）
        current_page_files, start_idx, end_idx = get_current_page_paths(
            selected_files, page_number, samples_per_page
        )

        st.write(f"显示 {start_idx+1} 到 {end_idx} 个样本，共 {len(selected_files)} 个")

        # 批量获取当前页视频的信息
        video_info_dict = batch_get_video_info(current_page_files)

        # 如果使用多进程，一次性批量生成当前页面所有视频的GIF
        gif_data_dict = {}
        if use_multiprocessing:
            with st.spinner("正在使用多进程并行生成所有GIF预览..."):
                gif_data_dict = batch_convert_videos_to_gifs(
                    current_page_files,
                    max_frames=gif_max_frames,
                    fps=gif_fps,
                    resize_factor=gif_resize_factor
                )

        # 使用列布局显示样本
        num_samples = len(current_page_files)
        num_rows = (num_samples + samples_per_row - 1) // samples_per_row

        # 显示样本网格
        for row in range(num_rows):
            cols = st.columns(samples_per_row)
            for col in range(samples_per_row):
                sample_idx = row * samples_per_row + col
                if sample_idx < num_samples:
                    video_path = current_page_files[sample_idx]

                    with cols[col]:
                        st.subheader(f"样本 #{start_idx + sample_idx + 1}")
                        st.caption(f"文件: {os.path.basename(video_path)}")

                        # 检查视频文件是否存在
                        if os.path.exists(video_path):

                            # 显示GIF
                            if use_multiprocessing:
                                # 从之前批处理的结果中获取数据
                                if video_path in gif_data_dict and gif_data_dict[video_path]:
                                    render_gif_html(gif_data_dict[video_path])
                                else:
                                    st.error("无法生成GIF")
                            else:
                                # 单独处理每个视频
                                with st.spinner(f"生成GIF预览 #{start_idx + sample_idx + 1}..."):
                                    # 调用转换为GIF的工作函数
                                    video_path, gif_data = _convert_video_to_gif_worker(
                                        video_path,
                                        max_frames=gif_max_frames,
                                        fps=gif_fps,
                                        resize_factor=gif_resize_factor
                                    )

                                    if gif_data:
                                        render_gif_html(gif_data)
                                    else:
                                        st.error("无法生成GIF")


                        else:
                            st.error(f"视频文件不存在: {video_path}")

    except Exception as e:
        import traceback
        st.error(f"处理过程中出错: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
