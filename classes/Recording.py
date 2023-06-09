# Recording.py

"""Recording class with variables and methods for working with a single recodring."""

import RecordingUtil as ru


class Recording:
    def __int__(self, recording_path: str):
        """
        Initialise a Recording object using the given path string.

        :param recording_path: Path to Recording directory as a String.
        """
        # Path to Recording directory.
        self.path = recording_path
        # Recording frames, stored in working memory.
        self.frames = ru.load_frames(self.path)
        # Current frame being displayed
        self.current_frame = 1
