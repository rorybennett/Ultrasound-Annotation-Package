import os


def load_frames(recording_path: str):
    """
    Load all .png images in recording_path as frames into a multidimensional list/

    :param recording_path: String representation of the recording path.

    :return frames: List of all the .png frames saved in the recording path directory.
    """
    contents = os.listdir(recording_path)

    frames = []


