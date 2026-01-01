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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video clustering sample visualization tool")
    parser.add_argument("--index_file", type=str, required=True, help="Clustering result file path (e.g. inverted_index.pkl)")
    parser.add_argument("--video_list", type=str, required=True, help="Video file list path")
    return parser.parse_args()

# Set page configuration
st.set_page_config(
    page_title="Video Clustering Sample Visualization Tool",
    layout="wide",
    menu_items={
        'About': "# Video Clustering Sample Visualization Tool\nVideo clustering data visualization application built with Streamlit"
    }
)

# Use caching mechanism to cache file list
@st.cache_resource
def load_video_list(list_path: str) -> List[str]:
    """
    Load video file list

    Args:
        list_path: Video list file path

    Returns:
        Video file path list
    """
    try:
        print(f"Loading video list: {list_path}")
        video_paths = []

        with open(list_path, 'r') as f:
            for line in f:
                video_path = line.strip()
                if video_path:  # Ignore empty lines
                    video_paths.append(video_path)

        print(f"Successfully loaded video list, containing {len(video_paths)} videos")
        return video_paths
    except Exception as e:
        st.error(f"Error loading video list: {str(e)}")
        return []

# Use caching mechanism to cache inverted index
@st.cache_resource
def load_inverted_index(index_path: str) -> Dict:
    """
    Load inverted index (stores only indices not actual paths for better performance)

    Args:
        index_path: Index file path

    Returns:
        Inverted index dictionary, each cluster corresponds to a set of video indices
    """
    try:
        print(f"Loading inverted index: {index_path}")
        if index_path.endswith('.pkl'):
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                print(f"Successfully loaded pickle file, containing {len(data)} clusters")
                return data
        elif index_path.endswith('.json'):
            import json
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded JSON file, containing {len(data)} clusters")
                return data
        else:
            st.error(f"Unsupported index file format: {index_path}")
            return {}
    except Exception as e:
        import traceback
        st.error(f"Error loading inverted index: {str(e)}")
        st.code(traceback.format_exc())
        return {}

def get_paths_for_cluster(cluster_indices: List[int], video_list: List[str]) -> List[str]:
    """
    Convert indices to paths only for the currently selected cluster

    Args:
        cluster_indices: List of video indices in the cluster
        video_list: List of all video paths

    Returns:
        List of video paths in the cluster
    """
    valid_paths = []
    for idx in cluster_indices:
        if 0 <= idx < len(video_list):
            valid_paths.append(video_list[idx])
    return valid_paths

@st.cache_data
def extract_frame_from_video(video_path: str, frame_index: int = 0) -> Optional[bytes]:
    """
    Extract a specific frame from video as thumbnail (using st.cache_data for better performance)

    Args:
        video_path: Video file path
        frame_index: Frame index to extract

    Returns:
        Binary data of the frame image
    """
    try:
        if not os.path.exists(video_path):
            return None

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Check if successfully opened
        if not cap.isOpened():
            return None

        # Get total number of frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return None

        # Adjust frame index to ensure validity
        frame_index = min(frame_index, total_frames - 1)

        # Set video position to specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read frame
        ret, frame = cap.read()

        # Release video
        cap.release()

        if not ret or frame is None:
            return None

        # Convert to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)

        return buffer.tobytes()

    except Exception as e:
        print(f"Error extracting video frame: {str(e)}")
        return None

# Non-cached GIF conversion function for multiprocessing calls
def _convert_video_to_gif_worker(video_path: str, max_frames: int = 50, fps: int = 10,
                                resize_factor: float = 0.5) -> Optional[Tuple[str, str]]:
    """
    Worker function to convert video to GIF format (for multiprocessing)

    Args:
        video_path: Video file path
        max_frames: Maximum number of frames to limit GIF size
        fps: Frames per second
        resize_factor: Resize factor (0.5 means half of original size)

    Returns:
        (video_path, Base64-encoded GIF data) tuple
    """
    try:
        if not os.path.exists(video_path):
            return (video_path, None)

        # Create temporary file to save GIF
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
            gif_path = tmp_file.name

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Check if successfully opened
        if not cap.isOpened():
            return (video_path, None)

        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)

        # Calculate sampling interval to ensure not exceeding max_frames
        sampling_interval = max(1, total_frames // max_frames)

        # Collect frames
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only collect every sampling_interval frame
            if frame_count % sampling_interval == 0:
                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))
                # Convert color space
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

            frame_count += 1

            # Limit maximum number of frames
            if len(frames) >= max_frames:
                break

        # Release video
        cap.release()

        if not frames:
            return (video_path, None)

        # Generate GIF
        imageio.mimsave(gif_path, frames, fps=fps, format='GIF')

        # Read GIF file and convert to base64
        with open(gif_path, 'rb') as f:
            gif_data = f.read()

        # Delete temporary file
        os.unlink(gif_path)

        # Convert to base64
        b64_gif = base64.b64encode(gif_data).decode('utf-8')

        # Return data URI
        return (video_path, f"data:image/gif;base64,{b64_gif}")

    except Exception as e:
        import traceback
        print(f"Error converting video to GIF ({video_path}): {str(e)}")
        print(traceback.format_exc())
        return (video_path, None)

# Use multiprocessing to batch convert videos to GIF
def batch_convert_videos_to_gifs(video_paths: List[str], max_frames: int = 50,
                               fps: int = 10, resize_factor: float = 0.5) -> Dict[str, str]:
    """
    Use multiprocessing to convert multiple videos to GIF in parallel

    Args:
        video_paths: List of video paths
        max_frames: Maximum number of frames
        fps: Frame rate
        resize_factor: Scale ratio

    Returns:
        Dictionary with video paths as keys and corresponding GIF data URIs as values
    """
    # Check if video path list is empty
    if not video_paths:
        return {}

    # Create a process pool with number of processes equal to CPU cores or video count (whichever is smaller)
    num_processes = min(multiprocessing.cpu_count(), len(video_paths))

    # Show progress
    start_time = time.time()
    st.write(f"Processing {len(video_paths)} videos in parallel using {num_processes} processes...")

    # Prepare conversion function
    convert_func = partial(
        _convert_video_to_gif_worker,
        max_frames=max_frames,
        fps=fps,
        resize_factor=resize_factor
    )

    results = {}

    try:
        # Use process pool to process videos in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Start async tasks
            result_objects = [
                pool.apply_async(convert_func, (video_path,))
                for video_path in video_paths
            ]

            # Prepare progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # Collect results
            for i, result_obj in enumerate(result_objects):
                # Update progress bar
                progress = (i + 1) / len(result_objects)
                progress_bar.progress(progress)
                progress_text.text(f"Processing progress: {i+1}/{len(result_objects)} ({progress*100:.1f}%)")

                # Get results
                try:
                    video_path, gif_data = result_obj.get(timeout=30)  # Set timeout
                    if gif_data:
                        results[video_path] = gif_data
                except Exception as e:
                    print(f"Error getting GIF conversion result: {str(e)}")

        # Clean up progress display
        progress_bar.empty()
        progress_text.empty()

        # Show processing time
        end_time = time.time()
        st.write(f"GIF processing complete, took {end_time - start_time:.2f} seconds")

    except Exception as e:
        import traceback
        print(f"Multiprocessing video conversion failed: {str(e)}")
        print(traceback.format_exc())

    return results

# Get video basic information
@st.cache_data
def get_video_info(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    """
    Get video basic information

    Args:
        video_path: Video file path

    Returns:
        (total_frames, fps, (width, height)) tuple
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
        print(f"Error getting video info: {str(e)}")
        return (0, 0.0, (0, 0))

# Batch get information for multiple videos
def batch_get_video_info(video_paths: List[str]) -> Dict[str, Tuple[int, float, Tuple[int, int]]]:
    """
    Get basic information for multiple videos

    Args:
        video_paths: List of video paths

    Returns:
        Dictionary with video paths as keys and video info tuples as values
    """
    results = {}
    for path in video_paths:
        results[path] = get_video_info(path)
    return results

# Optimization: get cluster path list without loading actual videos
def get_cluster_paths_only(clusters, selected_cluster_idx):
    """
    Get video paths for selected cluster

    Args:
        clusters: List of all clusters
        selected_cluster_idx: Selected cluster index

    Returns:
        Label and video path list of selected cluster
    """
    selected_label, selected_files = clusters[selected_cluster_idx]
    return selected_label, selected_files

# Optimization: get video paths for current page only
def get_current_page_paths(selected_files, page_number, samples_per_page):
    """
    Get video paths for current page

    Args:
        selected_files: All video paths of selected cluster
        page_number: Current page number
        samples_per_page: Number of samples per page

    Returns:
        Video paths for current page, start index and end index
    """
    start_idx = (page_number - 1) * samples_per_page
    end_idx = min(start_idx + samples_per_page, len(selected_files))
    current_page_files = selected_files[start_idx:end_idx]
    return current_page_files, start_idx, end_idx

def render_gif_html(gif_data_uri: str, width: str = "100%") -> None:
    """
    Render HTML code for GIF

    Args:
        gif_data_uri: Data URI containing GIF data
        width: GIF width

    Returns:
        None
    """
    html_code = f"""
    <img src="{gif_data_uri}" width="{width}" style="display:block; margin:auto;">
    """
    st.markdown(html_code, unsafe_allow_html=True)

# Streamlit application
def main():
    # Get command line arguments
    try:
        args = parse_arguments()
        index_file = args.index_file
        video_list_file = args.video_list
    except SystemExit:
        # When running in Streamlit, cannot parse command line arguments, use default values below
        # These values can be overridden by Streamlit command line options
        if 'index_file' in st.session_state and 'video_list_file' in st.session_state:
            index_file = st.session_state.index_file
            video_list_file = st.session_state.video_list_file
        else:
            # Allow user input on first run
            st.title("Video Clustering Sample Visualization Tool")
            st.write("Please enter the necessary file paths:")

            # Get file paths
            index_file = st.text_input("Clustering result file path (inverted_index.pkl):", "/path/to/inverted_index.pkl")
            video_list_file = st.text_input("Video list file path:", "/path/to/video_list.txt")

            if st.button("Confirm"):
                if index_file and video_list_file:
                    # Save paths to session state
                    st.session_state.index_file = index_file
                    st.session_state.video_list_file = video_list_file
                    st.rerun()
                else:
                    st.error("Please enter all necessary file paths")

            # Only show input form on first run
            return

    st.title("Video Clustering Sample Visualization Tool")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Show currently used files
    st.sidebar.subheader("Current Files")
    st.sidebar.text(f"Clustering file: {os.path.basename(index_file)}")
    st.sidebar.text(f"Video list: {os.path.basename(video_list_file)}")

    # Modify file paths
    if st.sidebar.button("Modify File Paths"):
        # Clear file paths from session state
        if 'index_file' in st.session_state:
            del st.session_state.index_file
        if 'video_list_file' in st.session_state:
            del st.session_state.video_list_file
        st.rerun()

    # Configuration parameters
    st.sidebar.subheader("Visualization Configuration")

    # Number of samples per page
    samples_per_page = st.sidebar.number_input("Samples per page", min_value=1, max_value=20, value=8)

    # Number of samples per row
    samples_per_row = st.sidebar.number_input("Samples per row", min_value=1, max_value=4, value=2)

    # GIF related configuration
    st.sidebar.subheader("GIF Configuration")

    # GIF frame rate
    gif_fps = st.sidebar.slider("GIF frame rate", min_value=1, max_value=30, value=10)

    # GIF maximum frames
    gif_max_frames = st.sidebar.slider("GIF max frames", min_value=10, max_value=100, value=50)

    # GIF size scale factor
    gif_resize_factor = st.sidebar.slider("GIF scale ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    # Multiprocessing configuration
    st.sidebar.subheader("Multiprocessing Configuration")
    use_multiprocessing = st.sidebar.checkbox("Use multiprocessing to speed up GIF generation", value=True)

    # Process inverted index
    try:
        # Show selected file paths
        st.write(f"Clustering file: **{os.path.basename(index_file)}** ({index_file})")
        st.write(f"Video list: **{os.path.basename(video_list_file)}** ({video_list_file})")

        # Check if file exists
        if not os.path.exists(index_file):
            st.error(f"Clustering file does not exist: {index_file}")
            return

        if not os.path.exists(video_list_file):
            st.error(f"Video list file does not exist: {video_list_file}")
            return

        # Load video list
        with st.spinner("Loading video list..."):
            video_list = load_video_list(video_list_file)
            if not video_list:
                st.error("Video list is empty or cannot be loaded")
                return

        # Load inverted index (read only once, will be cached)
        with st.spinner("Loading clustering data..."):
            inverted_index = load_inverted_index(index_file)

            if not inverted_index:
                st.error("Clustering file is empty or format error")
                return

        # Sort by cluster size
        clusters = [(label, indices) for label, indices in inverted_index.items()]
        clusters.sort(key=lambda x: len(x[1]), reverse=True)

        # Show cluster information
        st.header("Clusters Sorted by Size")

        # Initialize random idx (if not set yet)
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = random.randint(0, len(clusters) - 1)

        # Show current cluster info as hint
        st.write(f"Total of {len(clusters)} clusters, range is 0-{len(clusters)-1}")

        # Use number_input instead of selectbox, and set initial value to random number
        selected_cluster_idx = st.number_input(
            "Enter cluster index to view",
            min_value=0,
            max_value=len(clusters)-1,
            value=st.session_state.random_idx,
            step=1
        )

        # Get paths for selected cluster (without loading videos)
        selected_label, selected_indices = clusters[selected_cluster_idx]
        selected_files = get_paths_for_cluster(selected_indices, video_list)
        st.write(f"Selected cluster: {selected_label}, containing {len(selected_files)} samples")

        # Randomly sort sample list
        if 'sorted_files_key' not in st.session_state or st.session_state.sorted_files_key != selected_cluster_idx:
            random.seed(42)  # Use fixed seed to ensure consistent sorting result each time
            random.shuffle(selected_files)
            st.session_state.sorted_files_key = selected_cluster_idx

        # Pagination control
        total_pages = max(1, (len(selected_files) + samples_per_page - 1) // samples_per_page)
        page_number = st.number_input("Page number", min_value=1, max_value=total_pages, value=1)

        # Get video paths for current page (without loading videos)
        current_page_files, start_idx, end_idx = get_current_page_paths(
            selected_files, page_number, samples_per_page
        )

        st.write(f"Showing samples {start_idx+1} to {end_idx}, total {len(selected_files)}")

        # Batch get info for current page videos
        video_info_dict = batch_get_video_info(current_page_files)

        # If using multiprocessing, batch generate GIFs for all videos on current page at once
        gif_data_dict = {}
        if use_multiprocessing:
            with st.spinner("Generating all GIF previews in parallel using multiprocessing..."):
                gif_data_dict = batch_convert_videos_to_gifs(
                    current_page_files,
                    max_frames=gif_max_frames,
                    fps=gif_fps,
                    resize_factor=gif_resize_factor
                )

        # Use column layout to display samples
        num_samples = len(current_page_files)
        num_rows = (num_samples + samples_per_row - 1) // samples_per_row

        # Display sample grid
        for row in range(num_rows):
            cols = st.columns(samples_per_row)
            for col in range(samples_per_row):
                sample_idx = row * samples_per_row + col
                if sample_idx < num_samples:
                    video_path = current_page_files[sample_idx]

                    with cols[col]:
                        st.subheader(f"Sample #{start_idx + sample_idx + 1}")
                        st.caption(f"File: {os.path.basename(video_path)}")

                        # Check if video file exists
                        if os.path.exists(video_path):

                            # Display GIF
                            if use_multiprocessing:
                                # Get data from previous batch processing results
                                if video_path in gif_data_dict and gif_data_dict[video_path]:
                                    render_gif_html(gif_data_dict[video_path])
                                else:
                                    st.error("Unable to generate GIF")
                            else:
                                # Process each video individually
                                with st.spinner(f"Generating GIF preview #{start_idx + sample_idx + 1}..."):
                                    # Call the GIF conversion worker function
                                    video_path, gif_data = _convert_video_to_gif_worker(
                                        video_path,
                                        max_frames=gif_max_frames,
                                        fps=gif_fps,
                                        resize_factor=gif_resize_factor
                                    )

                                    if gif_data:
                                        render_gif_html(gif_data)
                                    else:
                                        st.error("Unable to generate GIF")


                        else:
                            st.error(f"Video file does not exist: {video_path}")

    except Exception as e:
        import traceback
        st.error(f"Error during processing: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
