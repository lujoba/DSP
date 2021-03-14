'''
feature extraction module.
Create by Luiz Barbosa 2018

Exemple usage:

import features.py as features

features_extractor = features.features([features.MAV(),
                                        features.VAR(),
                                        features.WL()],
                                        number_of_channels)

features_extractor.process(data)
'''

import numpy as np
import scipy.signal as sc # Added 04/09/2019

class features_extractor(object):

    def __init__(self, features, n_ch):
        super(features_extractor, self).__init__()
        self.features = features
        self.n_ch = n_ch
        self.n_features = n_ch * sum(
            [f.dim_per_channel for f in self.features])
        self.output = np.zeros(self.n_ch * self.n_features)

    def process(self, data):
        return np.hstack([f.compute(data) for f in self.features])

    def __repr__(self):
        return "%s.%s(%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            str([str(f) for f in self.features])
        )

class feature(object):

    def __repr__(self):
        return "%s.%s()" % (
            self.__class__.__module__,
            self.__class__.__name__
        )

class MAV(feature):
    """
    Calculates the mean absolute value of a signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.mean(np.absolute(x), axis=0)

        return y

class WL(feature):
    """
    Calculates the waveform length of a signal. Waveform length is just the
    sum of the absolute value of all deltas (between adjacent taps) of a
    signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.sum(np.absolute(np.diff(x)), axis=0)

        return y

class ZC(feature):
    """
    Calculates the number of zero crossings in a signal, subject to a threshold
    for discarding noisy fluctuations above and below zero.

    Parameters
    ----------
    thresh : float (default=0.0)
        The threshold for discriminating true zero crossings from those caused
        by noise.
    use_sm : bool (default=False)
        Specifies if spectral moments should be used for the computation. This
        is much faster, but the threshold is not taken into account, making it
        potentially affected by noise.
    """

    def __init__(self, thresh=0.0, use_sm=False):
        self.dim_per_channel = 1
        self.thresh = thresh
        self.use_sm = use_sm


    def compute(self, x):
        if self.use_sm:
            y = np.sqrt(
                SpectralMoment(2).compute(x) / SpectralMoment(0).compute(x))

        else:
            xrows, xcols = x.shape
            y = np.zeros(xcols)
            for i in range(xcols):
                for j in range(1, xrows):
                    if ((x[j, i] > 0 and x[j-1, i] < 0) or
                            (x[j, i] < 0 and x[j-1, i] > 0)):
                        if np.absolute(x[j, i] - x[j-1, i]) > self.thresh:
                            y[i] += 1

        return y

class SSC(feature):
    """
    Calculates the number of slope sign changes in a signal, subject to a
    threshold for discarding noisy fluctuations.

    Parameters
    ----------
    thresh : float (default=0.0)
        The threshold for discriminating true slope sign changes from those
        caused by noise.
    use_sm : bool (deafult=False)
        Specifies if spectral moments should be used for the computation. This
        is much faster, but the threshold is not taken into account, making it
        potentially affected by noise.
    """


    def __init__(self, thresh=0.0, use_sm=False):
        self.dim_per_channel = 1
        self.thresh = thresh
        self.use_sm = use_sm

    def compute(self, x):
        if self.use_sm:
            y = np.sqrt(
                SpectralMoment(4).compute(x) / SpectralMoment(2).compute(x))

        else:
            xrows, xcols = x.shape
            y = np.zeros(xcols)
            for i in range(xcols):
                for j in range(1, xrows-1):
                    if ((x[j, i] > x[j-1, i] and x[j, i] > x[j+1, i]) or
                            (x[j, i] < x[j-1, i] and x[j, i] < x[j+1, i])):
                        if (np.absolute(x[j, i]-x[j-1, i]) > self.thresh or
                                np.absolute(x[j, i]-x[j+1, i]) > self.thresh):
                            y[i] += 1
        return y

class SpectralMoment(feature):
    """
    Calculates the nth-order spectral moment.

    Parameters
    ----------
    n : int
        The spectral moment order. Should be even and greater than or equal to
        zero.
    """

    def __init__(self, n):
        self.dim_per_channel = 1
        self.n = n

    def compute(self, x):
        xrows, xcols = x.shape
        y = np.zeros(xcols)

        if self.n % 2 != 0:
            
            return y

        # special case, zeroth order moment is just the power
        if self.n == 0:
            y = np.sum(np.multiply(x, x), axis=0)

        else:
            y = SpectralMoment(0).compute(np.diff(x, int(self.n/2), axis=0))

        return np.mean(y)

class KhushabaSet(feature):
    """
    Calcuates a set of 5 features introduced by Khushaba et al. at ISCIT 2012.
    (see reference [1]). They are:
        1. log of the 0th order spectral moment
        2. log of normalized 2nd order spectral moment (m2 / m0^u)
        3. log of normalized 4th order spectral moment (m4 / m0^(u+2))
        4. log of the sparseness (see paper)
        5. log of the irregularity factor / waveform length (see paper)

    Parameters
    ----------
    u : int (default=0)
        Used in the exponent of m0 for normalizing higher-orer moments

    References
    ----------
    .. [1] `R. N. Khushaba, L. Shi, and S. Kodagoda, "Time-dependent spectral
        features for limb position invariant myoelectric pattern recognition,"
        Communications and Information Technologies (ISCIT), 2012 International
        Symposium on, 2012.`
    """

    def __init__(self, u=0):
        self.dim_per_channel = 5
        self.u = u


    def compute(self, x):
        xrows, xcols = x.shape
        # TODO fill this instead of using hstack
        # y = np.zeros(self.dim_per_channel*xcols)

        m0 = SpectralMoment(0).compute(x)
        m2 = SpectralMoment(2).compute(x)
        m4 = SpectralMoment(4).compute(x)
        S = m0 / np.sqrt(np.abs((m0-m2)*(m0-m4)))
        IF = np.sqrt(m2**2 / (m0*m4))

        return np.hstack((
            np.log(m0),
            np.log(m2 / m0**2),
            np.log(m4 / m0**4),
            np.log(S),
            np.log(IF / WL().compute(x))))

class SampEn(feature):
    """
    Calculates the sample entropy of time series data. See reference [1].


    The basic idea is to take all possible m-length subsequences of the
    time series and count the number of these subsequences whose Chebyshev
    distance from all other subsequences is less than the tolerance parameter,
    r (self-matches excluded). This is repeated for (m+1)-length subsequences,
    and SampEn is given by the log of the number of m-length matches divided
    by the number of (m+1)-length matches.

    This feature can have some issues if the tolerance r is too low and/or the
    subsequence length m is too high. A typical value for r is apparently
    0.2*std(x).

    Parameters
    ----------
    m : int
        Length of sequences to compare (>1)
    r : float
        Tolerance for counting matches.

    References
    ----------
    .. [1] `J. S. Richman and J. R. Moorman, "Physiological time series
        analysis using approximate entropy and sample entropy," American
        Journal of Physiology -- Heart and Circulatory Physiology, vol. 278
        no. 6, 2000.`
    """

    def __init__(self, m, r):
        self.dim_per_channel = 1
        self.m = m
        self.r = r

    def compute(self, x):
        xrows, xcols = x.shape
        y = np.zeros(xcols)
        m = self.m
        N = xrows

        for c in range(xcols):
            correl = np.zeros(2) + np.finfo(np.float).eps

            xmat = np.zeros((m+1, N-m+1))
            for i in range(m):
                xmat[i, :] = x[i:N-m+i+1, c]
            # handle last row separately
            xmat[m, :-1] = x[m:N, c]
            xmat[-1, -1] = 10*np.max(xmat)  # something that won't get matched

            for mc in [m, m+1]:
                count = 0
                for i in range(N-mc-1):
                    dist = np.max(
                        np.abs(xmat[:mc, i+1:] - xmat[:mc, i][:, np.newaxis]),
                        axis=0)

                    count += np.sum(dist <= self.r)

                correl[mc-m] = count
                
            y[c] = np.log(correl[0] / correl[1])


        return y

### Added 04/09/2019

class MV(feature):
    """
    Calculates the mean value of a signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.mean(x, axis=0)

        return y

class MD(feature):
    """
    Calculates the median value of a signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.median(x, axis=0)

        return y

class VAR(feature):
    """
    Calculates the mean absolute value of a signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.var(x, axis=0)

        return y

class STD(feature):
    """
    Calculates the standart deviation of the value of a signal.
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.std(x, axis=0)

        return y

class RMS(feature):
    """
    Calcuates the RMS of the value of the signal
    """

    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.sqrt(np.mean(np.square(x), axis=0))

        return y

class PKS(feature):
   """
   Calcuates the Peaks using RMS of the value of the signal
   """

   def __init__(self):
       self.dim_per_channel = 1

   def compute(self, x):
       rms = RMS().compute(x)
       pks = np.array([sc.find_peaks_cwt(window, np.ones(len(x)), min_length = rms) for channel in x for window in channel])
       y = [channel[[pks]] for channel in x]
        
       return y

class MPKS(feature):
    """
    Calcuates the Peaks using RMS of the value of the signal
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        pks = PKS().compute(x)
        y = [channel[pks] for channel in x]
        y = np.mean(y, axis=0)

        return y

class MVEL(feature):
    """
    Mean firing velocity using the peaks
    """
    
    """
    if ~isfield(pF.f,'trms')
        pF = GetSigFeatures_trms(pF);
    end
    if ~isfield(pF,'pks')        
        for i = 1 : pF.ch
            [pks, locs] = findpeaks(pF.absdata(:,i),'minpeakheight',pF.f.trms(i));
            pF.pks{i} = pks;
            pF.locs{i} = locs;
        end
    end        
    if ~isfield(pF,'pksData')
        for i = 1 : pF.ch
            pF.pksData{i} = pF.data(pF.locs{1}, i);          % Only data from the peaks
            pF.diffPksData{i} = diff(pF.pksData{i});           % Get the diff of the peaks or velocity of the peaks
        end
    end    

    for i = 1 : pF.ch
        if isempty(pF.diffPksData{i})
            pF.f.tmvel(i) = 0;
        else
            pF.f.tmvel(i) = mean(abs(pF.diffPksData{i})); % Get mean of the firing velocity
        end
    end
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        return

class PWR(feature):
    """
    Power of the signal
    """
    def __init__(self, sf):
        self.dim_per_channel = 1
        self.sf = sf

    def compute(self, x):
        y = np.sum((x*x)/(self.sf),axis=0)

        return y

class CR(feature):
    """
    Correlation:
    * Close to 1 means mutual increment
    * Close to -1 means mutual decrement
    * Close to 0 is no correlation or no-linear correlation
    """
    def __init__(self, ch):
        self.dim_per_channel = 1
        self.ch = ch

    def compute(self, x):
        
        for i in range(self.dim_per_channel):
            j = i + 1
            for k in range(j): 
                y = np.corrcoef(j, k)

        return y

# class CV(feature):
#     """
#     Covariance
#     * Note: It is possible that the covariance is not required because corr is
#     * computer already
#     """
#     def __init__(self):
#         self.dim_per_channel = 1

#     def compute(self, x):
#         mcr = cov(pF.data);
#         k=1;
#         for i = 1: pF.ch
#             for j = i+1 : pF.ch
#                 pF.f.tcv(k) = mcr(i,j);
#                 k=k+1;

class FD(feature):
    """
    % Find the fractal dimension
    """
    def __init__(self, thresh=0.9):
        self.dim_per_channel = 1
        self.thresh = thresh

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(self, x, k):
        S = np.add.reduceat(
            np.add.reduceat(x, np.arange(0, x.shape[0], k), axis=0),
                               np.arange(0, x.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
        
    def compute(self, x):
        # Transform x into a binary array
        x = (x < self.thresh)

        # Minimal dimension of image
        p = min(x.shape)

        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))

        # Extract the exponent
        n = int(np.log(n)/np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(self.boxcount(x, size))

        # Fit the successive log(sizes) with log (counts)
        y = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -y[0]

class ADiff(feature):
    """
    Find the absolute diference
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.sum(np.mean(0 - x), axis=0)

        return y

class ASS(feature):
    """
    The Absolute value of the Summation of the Square root (ASS)
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.sum(np.sqrt(np.abs(x)), axis=0)

        return y

class MSR(feature):
    """
    The Mean value of the Square Root (MSR): It provides an estimated 
    measure of the total amount of activity in the analysis window.
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        y = np.sum(np.mean(np.sqrt(np.abs(x)), axis=0))

        return y
        
class ASM(feature):
    """
    The Absolute value of the Summation of the exp th root (ASM): It provides
    a comprehensive insight into the amplitude of the EMG signal since it
    gives an approximate measure of the power of the signal which also produces
    a waveform that is easily analyzable. The exp. variable can assume one of 
    two possible values (0.50 or 0.75) based on the characteristic of the signal
    segment under analysis.
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        for i in range (len(x)):
            if i <= (0.25 * len(x)) or i >= (0.75 * len(x)):
                y = np.sum((np.abs(x)**0.5), axis=0)
            else:
                y = np.sum((np.abs(x)**0.75), axis=0)

        return y
        
class LDR:
    """
    Levinson Durbin Recursion
    --------|---------------------
    R_k     | toeplitz_elements[k]
    --------|---------------------
    E_k     | extra_element
    --------|---------------------
    A_k     | solutions[k]
    --------|---------------------
    U_k     | extended_solution[k]
    --------|---------------------------
    V_k     | r_extended_solution[k]
    Reference: http://www.emptyloop.com/technotes/A%20tutorial%20on%20linear%20prediction%20and%20Levinson-Durbin.pdf
    """

    def __init__(self, toeplitz_elements):
        self.toeplitz_elements = toeplitz_elements
        self.lpc_dim = len(toeplitz_elements) - 2
        self.dim_per_channel = self.lpc_dim

    def compute(self):
        solutions, extra_ele = self.solve_size_one()
        y, _ = self.solve_recursively(solutions, extra_ele)

        return y

    def solve_size_one(self):
        solutions = np.array([1.0, - self.toeplitz_elements[1] / self.toeplitz_elements[0]])
        extra_element = self.toeplitz_elements[0] + self.toeplitz_elements[1] * solutions[1]

        return solutions, extra_element

    def solve_recursively(self, initial_solutions, initial_extra_ele):
        extra_element = initial_extra_ele
        solutions = initial_solutions

        for k in range(1, self.lpc_dim):
            lambda_value = self._calculate_lambda(k, solutions, extra_element)
            extended_solution = self._extend_solution(solutions)
            r_extended_solution = extended_solution[::-1]

            solutions = extended_solution + lambda_value * r_extended_solution
            extra_element = self._calculate_extra_element(extra_element, lambda_value)

        return solutions, extra_element

    def _extend_solution(self, previous_solution):

        return np.hstack((previous_solution, np.array([0.0])))

    def _calculate_extra_element(self, previous_extra_ele, lambda_value):

        return (1.0 - lambda_value**2) * previous_extra_ele

    def _calculate_lambda(self, k, solutions, extra_element):
        tmp_value = 0.0

        for j in range(0, k + 1):
            tmp_value += (- solutions[j] * self.toeplitz_elements[k + 1 - j])

        return tmp_value / extra_element
        
"""
function pF = GetSigFeatures_tdam(pF)

    temp = zeros(size(pF.data));
    temp(1:end-1,:) = pF.data(2:end,:); % Compute k+1
    diffAmp = abs(temp - pF.data);      % compute the absulute difference
    pF.f.tdam = sum(diffAmp(1:end-1,:)) ./ (pF.sp-1); 

end

function pF = GetSigFeatures_tfd(pF)

    mdata = [zeros(1,pF.ch) ; pF.data(1:pF.sp-1,:)];
    absDiff = abs(pF.data - mdata); 
    L = sum(absDiff);    
    n = pF.sp;  %Data points or samples
    % This is not calculated properly, max of absDiff is only the max
    % distance between adjacent points and not all the points in the set
    d = max(absDiff); % Max distance between two points.
    
    pF.f.tfd = log(n) ./ (log(n) + (d./L));
    
end

function pF = GetSigFeatures_tmfl(pF)
% Maximum Fractal Length

    N = pF.sp;  %Total samples
    m = 1;      %Initial time
    
    for k = 1 : 9 : 10;
        limTop = floor((N-m)/k);
        for i = 1 : limTop
            a = m+(i*k);
            b = m+((i-1)*k);
            tempL(i,:) = abs(pF.data(a,:) - pF.data(b,:));
        end
        L(k,:) = sum(tempL) .* ((N-1)/(limTop*k)) ./ k; 
        clear tempL;
    end

    pF.f.tmfl = L(1,:);
    
    pF.L = L;

end

function pF = GetSigFeatures_tfdh(pF)
% Fractal dimension using Higuchi algorithm

    if ~isfield(pF.f,'tmfl')
        pF = GetSigFeatures_tmfl(pF);
    end
    
    dX = log(pF.L(1,:))-log(pF.L(10,:));
    dY = log(10)-log(1);
    pF.f.tfdh = dX./dY;
    
end
"""
    
######################### Frequency Features #########################

class FFT(feature):
    """
    Compute the FFT of the signal
    """
    def __init__(self):
         self.dim_per_channel = 1

    def compute(self, x):
        y = np.fft.fft2(x)
        y = np.fft.fftshift(y)

        return y

class FWL(feature):
    """
    Waveform Length (acumulative changes in the length)
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        fft = FFT().compute(x)
        y = np.sum(np.absolute(fft), axis=0)

        return y

class FMN(feature):
    """
    Mean Frequency
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        fft = FFT().compute(x)
        y = np.mean(np.absolute(fft), axis=0)

        return y

class FMD(feature):
    """
    Median Frequency
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        fft = FFT().compute(x)
        y = np.median(np.absolute(fft), axis=0)

        return y

class FPMN(feature):
    """
    Find the highest frequency peaks and gets its mean
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        fft = FFT().compute(x)
        pks = PKS().compute(fft)
        y = [channel[pks] for channel in x]
        y = np.median(np.absolute(y), axis=0)

        return y

class FPMD(feature):
    """
    Find the highest frequency peaks and gets its mean
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x): 
        fft = FFT().compute(x)
        pks = PKS().compute(fft)
        y = [channel[pks] for channel in x]
        y = np.mean(np.absolute(y), axis=0)

        return y

class FPSTD(feature):
    """
    % Find the highest frequency peaks and gets its mean
    """
    def __init__(self):
        self.dim_per_channel = 1

    def compute(self, x):
        fft = FFT().compute(x)
        pks = PKS().compute(fft)
        y = [channel[pks] for channel in x]
        y = np.std(y, axis=0)

        return y
