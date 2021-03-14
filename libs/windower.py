from scipy import signal
import numpy as np

class windower(Object):
    """
    Takes new input data and combines with past data to maintain a sliding
    window with overlap. It is assumed that the input to this block has length
    (length-overlap).

    Parameters
    ----------
    length : int
        Total number of samples to output on each iteration.
    overlap : int, default=0
        Number of samples from previous input to keep in the current window.
    """

    def __init__(self, length, overlap=0):
        super(Windower, self).__init__()
        self.length = length
        self.overlap = overlap

        self.clear()

    def clear(self):
        self._out = None

    def process(self, data):
        if self._out is None:
            self._preallocate(data.shape[1])

        if self.overlap == 0:
            return data

        self._out[:self.overlap, :] = self._out[-self.overlap:, :]
        self._out[self.overlap:, :] = data

        return self._out.copy()

    def _preallocate(self, cols):
        self._out = np.zeros((self.length, cols))

    def __repr__(self):
        return "%s.%s(length=%s, overlap=%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.length,
            self.overlap
        )
