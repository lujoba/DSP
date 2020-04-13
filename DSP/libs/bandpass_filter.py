from scipy import signal
import numpy as np

class bandpass_filter(object):
    """
    Bandpass filters incoming data with a Butterworth filter.

    Parameters
    ----------
    order : int
        Filter order.
    fCut : tuple or list of ints (len=2)
        Cutoff frequencies for the filter (f_cut_low, f_cut_high).
    fSamp : int
        Sampling rate specified in the same units as the frequencies in f_cut.
    overlap : int (default=0)
        Number of samples overlapping in consecutive inputs. Needed for
        correct filter initial conditions in each filtering operation.
    """

    def __init__(self, order, fCut, fSamp, overlap = 0):
        super(bandpass_filter, self).__init__()
        self.order = order
        self.fCut = fCut
        self.fSamp = fSamp
        self.overlap = overlap
        self.build_filter()
        self.clear()

    def build_filter(self):
        wc = [f / (self.fSamp/2.0) for f in self.fCut]
        self.b, self.a = signal.butter(self.order, wc, 'bandpass')

    def clear(self):
        self.xPrev = None
        self.yPrev = None

    def process(self, data):
        if self.xPrev is None:
            # first pass has no initial conditions
            out = signal.lfilter(
                self.b, self.a, data, axis=0)
        else:
            # subsequent passes get ICs from previous input/output
            nCh = data.shape[1]
            K = max(len(self.a)-1, len(self.b)-1)
            self.zi = np.zeros((K, nCh))
            # unfortunately we have to get zi channel by channel
            for c in range(data.shape[1]):
                self.zi[:, c] = signal.lfiltic(
                    self.b,
                    self.a,
                    self.yPrev[-(self.overlap+1)::-1, c],
                    self.xPrev[-(self.overlap+1)::-1, c])

            out, zf = signal.lfilter(
                self.b, self.a, data, axis=0, zi=self.zi)

        self.xPrev = data
        self.yPrev = out

        return out

    def __repr__(self):
        return "%s.%s(order=%s, f_cut=%s, f_samp=%s, overlap=%d)" % (
            self.__class__.__module___,
            self.__class__.__name__,
            self.order,
            self.fCut,
            self.fSamp,
            self.overlap
        )