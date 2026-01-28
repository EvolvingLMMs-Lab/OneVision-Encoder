from . import api


def read_video(
        video_path: str,
        with_residual: bool = True
):
    """
    Read motion vector and residual from encoded video file
    :param video_path: Path to H.264 video file
    :param with_residual: Whether to read residual from video
    :return:
    """
    return api.read_video(
        video_path,
        int(not with_residual),  # without residual
    )
