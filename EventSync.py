"""Module containing function for syncing discrete events.

The Syncing Problem
---
Each set of timestamps is a list of events detected on two systems.
The underlying events are the same but 1) there is an unknown offset
in the clocks; 2) the clocks may run at a slightly different rate;
3) there is an unknown number of missing events in each list.

We know that there is a good linear fit between some values in
timestamps1 and some values in timestamps2. The problem could be
expressed as: what is the largest
set of one-to-one matched timestamps from each dataset that 
satisfy a pre-specified minimum R**2 value? 

The straightforward way to solve this is to create Gaussian smoothed
continuous versions of each event train, then find the best lag in
their cross-correlation. But this won't account for clock speed
differences, and also for long time ranges (1hr) and fine resolution
(10ms) it can take prohibitively long.

I don't know the analytical way to solve this so I use an iterative
process, which relies on the fact that the clock speed difference
is relatively small compared to the temporal delay.

Step 1) Estimate the gross delay. Gaussian smooth each list of
timestamps. Begin with a coarse resolution: wide Gaussians in time,
undersampled. Check correlation between signals at all lags. Choose
the best fit, and evaluate its likelihood by estimating the p-value of
observing such a lag. If it meets a threshold, continue to step 2.
Otherwise, sample more finely and continue.

Step 2) Iterative linear fit. Using the initial guess of delay from step
1 and assuming equal clock speeds, find the closest fit in set2 to each
element in set1. Discard any pair that is off by more than some parameter.
Linear regress the remaining pairs to get the fit for the next iteration.
Repeat until the number of matched pairs stops changing. Then halve
the error parameter and continue until a pre-specified error level is
reached.

To see debugging text or figures, set EventSync.EVENTSYNC_DEBUG or 
EventSync.EVENTSYNC_DEBUG_FIGURE to True

Current issues:
1) It takes a long time to come up with the initial guess for the fit,
especially if the assumed temporal dilation is slightly wrong. The problem
is identifying when the peak corr is "significantly good". Right now I approx
the distribution of corrs as normal and take the peak z-score p-value, corrected
for the number of points, but this is not a good estimate of this distribution
and therefore even good fits have bad p-values.
2) If the initial guess is not quite right, then the iterative fitting will
drop the points furthest out, then refit on remaining points, and so on,
continuing to drop points until very few are left.
Also, mismatch errors are not corrected. If the matching is wrong, the new
fit just reinforces this.
One way around this would be to do the linear fitting on all the
(non one-to-one) matches within error_term. In fact this is almost certainly
the right thing to do. Then stop descending when the fit is unique since we
won't get any better.

Short term fix: give it a good starting guess for temporal dilation and
all should be fine.

Potential longer term answers:
1) Estimate delay at scale S, then matching linfit at scale S, then use
the dilation and offset parameters at next lower scale S-1, etc. This is
probably okay as long as the starting scale is close enough to the right
scale so that the values are reasonable.
(Started this in sync2, works a little better but does not resolve
the matching problem.)
2) Search method ... search over ranges of delay and dilation and maximize
the number of points fit.
3) Perhaps combine the two. At each scale, estimate gross delay and then
define the search limits based on the scale. Then test all fits in the region
and choose best. Use that as the center point for the next smaller scale
At each scale, use the dilation from the previous linfit, and the intercept
from the previous acorr, since these methods are optimized for those params.

"""
from numpy import polyval, polyfit
from scipy.special import ndtr
import numpy as np
import sys
import PointProc

EVENTSYNC_DEBUG = False
EVENTSYNC_DEBUG_FIGURE = False


def invert_linear_poly(p):
    """Helper function for inverting fit.coeffs"""
    return np.array([1, -p[1]]).astype(np.float) / p[0]


class UnmatchedFit:
    """An object to hold links between two event trains.
    
    This object handles the bookkeeping for the best matching between
    two trains and provides convenient access functions for the mapping.
    
    My attributes
    x : Event train 1
    y : Event train 2 (not necessarily the same size as x)
    coeffs : a polynomial mapping from times in x to estimated times in y
    yhat : estimated times in y
    error_term : maximum cutoff for error between a value in y and its
        estimated value in yhat, to be considered the same point
    
    The update() method will estimate yhat, and choose an optimal one-to-one
    mapping from matched samples from x and y, such that the maximum error
    in y is less than error_term. If multiple pairs satisfy the constraint,
    the one that has the smallest error is chosen.
    
    xi2yi : dict mapping index into x to index into y
    xi2resid : dict mapping index into x to the prediction error
    yi2xi : inverse of xi2yi
    yi2resid : yi -> xi2resid[yi2xi[yi]]
    
    You can change x, y, error_term, or coeffs and then call update() again.
    The old map will be wiped and replaced with a new one.
    
    Some other convenient access properties
    xi2yi_table
    matched_xi
    etc
    
    len(obj) gives the number of matches. Note that this is usually less
    than len(obj.x) and len(obj.y), because not all data points will be
    matched (in general).
    """
    def __init__(self, x=None, y=None, coeffs=None, error_term=None):
        self.x = x
        self.y = y
        self.coeffs = coeffs
        self.error_term = error_term
        self.yhat = None
        self._init_map()
    
    def _init_map(self):
        self.xi2yi = {}
        self.yi2xi = {}
        self.xi2resid = {}
        self.yi2resid = {}

    def update(self, coeffs=None):
        """Update the internal matches and residuals based on coeffs.
        
        If coeffs not provided, uses stored self.coeffs
        """
        if coeffs is not None:
            self.coeffs = coeffs
        
        # Clear out old map
        self._init_map()
        
        # Evaluate prediction
        self.yhat = polyval(self.coeffs, self.x)
        
        # Find closest equivalents in y for each x and update the map
        for xxidx, (xx, yyhat) in enumerate(zip(self.x, self.yhat)):
            yyidx = np.argmin(np.abs(self.y - yyhat))
            resid = self.y[yyidx] - yyhat
            
            # This method will only update if it is a good fit
            self._update_map(xxidx, yyidx, resid)
    
    def __len__(self):
        return len(self.xi2yi)
    
    def diagnostic(self):
        mxi = self.matched_xi
        myi = self.matched_yi
        s = 'found %d matches, ' + \
            '%d/%d in x (min: %d, max: %d, inrange: %d), ' + \
            '%d/%d in y (min: %d, max: %d, inrange: %d)'
        print s % (len(self), 
            len(self.matched_x), len(self.x),
            mxi.min(), mxi.max(), mxi.max() - mxi.min() + 1,
            len(self.matched_y), len(self.y),
            myi.min(), myi.max(), myi.max() - myi.min() + 1)
    
    @property
    def xi2yi_table(self):
        # Consider memoizing this if running slowly
        table1 = np.array(self.xi2yi.items())
        table2 = np.array(self.yi2xi.items())
        table2 = table2[:, ::-1]
        
        # Sort by xi
        i1 = np.argsort(table1[:, 0])
        table1 = table1[i1, :]        
        i2 = np.argsort(table2[:, 0])
        table2 = table2[i2, :]
        
        assert np.all(table1 == table2)
        return table1
    
    @property
    def nan_x2y_table(self):
        """Like x2y_table, but with NaN for umatched entries"""
        pass
        res = []
        for xi in len(range(self.x)):
            if xi in fit.xi2yi:
                res.append(xi, fit.xi2yi[xi])
            else:
                res.append(None, )
            
    
    @property
    def yi2xi_table(self):
        return self.xi2yi_table[:, ::-1]
    
    @property
    def matched_xi(self):
        return self.xi2yi_table[:, 0]
    
    @property
    def matched_yi(self):
        return self.xi2yi_table[:, 1]
    
    @property
    def matched_x(self):
        return self.x[self.matched_xi]
    
    @property
    def matched_y(self):
        return self.y[self.matched_yi]
    
    @property
    def x2y_table(self):
        return np.hstack([self.matched_x[:, None], self.matched_y[:, None]])

    def _update_map(self, xi, yi, resid):
        """If xi<-->yi is a good match, store"""
        # Do nothing if doesn't satisfy condition
        if abs(resid) > self.error_term:
            return
        
        # Decide whether to store in map
        store_in_map = False
        if (xi not in self.xi2yi) and (yi not in self.yi2xi):
            # no competition, store in map
            store_in_map = True
        elif (xi in self.xi2yi) and (yi not in self.yi2xi):
            if abs(resid) < abs(self.xi2resid[xi]):
                # This is a better fit than the stored one
                # Break the stored link
                self.yi2resid.pop(self.xi2yi[xi])
                self.yi2xi.pop(self.xi2yi[xi])
                store_in_map = True
        elif (xi not in self.xi2yi) and (yi in self.yi2xi):
            if abs(resid) < abs(self.yi2resid[yi]):
                # This is a better fit than the stored one
                # Break the stored link
                self.xi2resid.pop(self.yi2xi[yi])
                self.xi2yi.pop(self.yi2xi[yi])
                store_in_map = True
        elif self.xi2yi[xi] == yi and self.yi2xi[yi] == xi:
            # this exact link is already stored
            pass
        else:
            # Both are already in the map, and they are out of sync
            raise ValueError("map out of sync")
        
        # Now actually update the dicts
        if store_in_map:
            self.xi2yi[xi] = yi
            self.xi2resid[xi] = resid            
            self.yi2xi[yi] = xi
            self.yi2resid[yi] = resid


def sync2(timestamps1, timestamps2, min_acceptable_error=.1,
    error_term_factor=1.5, gross_delay_scales=None,
    gross_delay_scales_start=None, gross_delay_scales_factor=2.0, n_scales=10,
    p_thresh=.05, clock_sync_guess=1.0, max_iter=100):
    """Top-level function to sync up two sets of timestamps.
    
    Step 1) Estimate gross delay with cross-correlation
    Step 2) Use this as the initial call to matching linfit. Set
    error term to the estimation error from Step 1.
        Step 2a) Update fit, count number of matches
        Step 2b) If number of matches changes, repeat 2a
    Step 3) If error term is less than global error term, stop. Otherwise,
    lower error term and return to step 2.
    """
    # Step 1
    # which scales to operate at
    if gross_delay_scales is None:        
        if gross_delay_scales_start is None:
            # Choose starting scale from data
            gross_delay_scales_start = np.mean([
                np.median(np.diff(sorted(timestamps1))),
                np.median(np.diff(sorted(timestamps2)))])
        gross_delay_scales = gross_delay_scales_start * (
            gross_delay_scales_factor ** (-np.arange(n_scales)))
    
    # Estimate gross delay: amount to delay ts2 to fit ts1
    dilation = clock_sync_guess
    for scale in gross_delay_scales:
        print scale
        gross_delay_samples, gross_delay_sec, gross_delay_error = \
            smooth_and_estimate_delay(
                dilation*timestamps1, timestamps2,
                scales=[scale], p_thresh=p_thresh)
        
        fit = UnmatchedFit(x=timestamps1, y=timestamps2, 
            error_term=gross_delay_error*2, coeffs=[dilation, -gross_delay_sec])
        
        fit.update()
        print fit.coeffs
        
        new_coeffs = np.polyfit(fit.matched_x, fit.matched_y, deg=1)
        print new_coeffs
        dilation = new_coeffs[0]
        fit.diagnostic()
    
    return fit

def sync(timestamps1, timestamps2, min_acceptable_error=.1,
    error_term_factor=1.5, gross_delay_scales=None,
    gross_delay_scales_start=None, gross_delay_scales_factor=2.0, n_scales=10,
    p_thresh=.05, clock_sync_guess=1.0, max_iter=100):
    """Top-level function to sync up two sets of timestamps.
    
    Step 1) Estimate gross delay with cross-correlation
    Step 2) Use this as the initial call to matching linfit. Set
    error term to the estimation error from Step 1.
        Step 2a) Update fit, count number of matches
        Step 2b) If number of matches changes, repeat 2a
    Step 3) If error term is less than global error term, stop. Otherwise,
    lower error term and return to step 2.
    """
    # Step 1
    # which scales to operate at
    if gross_delay_scales is None:        
        if gross_delay_scales_start is None:
            # Choose starting scale from data
            gross_delay_scales_start = np.mean([
                np.median(np.diff(sorted(timestamps1))),
                np.median(np.diff(sorted(timestamps2)))])
        gross_delay_scales = gross_delay_scales_start * (
            gross_delay_scales_factor ** (-np.arange(n_scales)))
    
    # Estimate gross delay: amount to delay ts2 to fit ts1
    gross_delay_samples, gross_delay_sec, gross_delay_error = \
        smooth_and_estimate_delay(
            clock_sync_guess*timestamps1, timestamps2,
            scales=gross_delay_scales, p_thresh=p_thresh)
    
    if EVENTSYNC_DEBUG:
        print "STEP 1: delay %0.3f at scale %0.3f" % (
            gross_delay_sec, gross_delay_error)
        sys.stdout.flush()

    # Step 2
    # Create fit object from data
    # Start with an error term equal to scale on which gross delay was found
    coeffs_guess = [clock_sync_guess, -gross_delay_sec]
    fit = UnmatchedFit(x=timestamps1, y=timestamps2, 
        error_term=gross_delay_error, coeffs=coeffs_guess)
    fit.update()
    if len(fit) <= 2:
        errmsg = ("Bad fit or insufficient data for step2, " + 
            "try lowering p_thresh or delay_scales")
        raise ValueError(errmsg)

    inum = 0
    while inum < max_iter:
        if EVENTSYNC_DEBUG:
            print "S i%d %0.5f %d" % (inum, fit.error_term, len(fit))
            print fit.coeffs
            sys.stdout.flush()
        inum += 1

        # Step 2a/2b inner loop inside this function
        fit = matching_linfit_loop_to_stable(fit)
        
        if fit.error_term > min_acceptable_error * 1.00001:
            # Set up next iteration
            fit.error_term = max(fit.error_term/error_term_factor, 
                min_acceptable_error)
        else:
            # fit found!
            if EVENTSYNC_DEBUG:
                print "S i%d %0.5f %d !" % (inum, fit.error_term, len(fit))
                print fit.coeffs     
                sys.stdout.flush()
            break
    
    if inum == max_iter:
        print "warning: max iterations exceed in `sync`"
    
    return fit
    
    
def matching_linfit_loop_to_stable(fit=None, x=None, y=None, error_term=1.0, 
    coeffs_guess=None, max_iter=100):    
    """Inner loop: for fixed error, find stable linfit"""
    # If no fit object provided, initialize from raw data
    if fit is None:
        fit = UnmatchedFit(x=x, y=y, error_term=error_term, coeffs=coeffs_guess)
    
    # Extract the current matches
    fit.update()    
    
    # Loop until stability is reached
    inum = 0
    current_table = fit.xi2yi_table.copy()
    while inum < max_iter:
        if EVENTSYNC_DEBUG:
            print "MLLTS i%d %d" % (inum, len(fit))
            print fit.coeffs
            sys.stdout.flush()

        # Update fit
        fit = matching_linfit(fit)
        
        if np.all(fit.xi2yi_table == current_table):
            # Stability reached
            if EVENTSYNC_DEBUG:
                print "MLLTS i%d %d !" % (inum, len(fit))
                print fit.coeffs     
                sys.stdout.flush()                
            break
    
        # Set up next iteration
        current_table = fit.xi2yi_table.copy()
        inum += 1
    
    if inum == max_iter:
        print "warning: matching_linfit_loop_to_stable did not converge"
    
    return fit

def matching_linfit(fit=None, x=None, y=None, error_term=1.0, coeffs_guess=None):
    """Match up almost-identical values in x and y and fit to line.
    
    fit : UnmatchedFit object, or None, in which case the following are used
        to created an UnmatchedFit object
    x   : raw timestamps
    y   : raw timestamps
    error_term : threshold to drop outliers
    coeffs_guess : initial guess for coefficients
    
    The fit object will be modified in place and returned.
    
    fit.update() is called to drop outliers and match x and y
    Then a new set of coefficients is calculated using the matched pairs.
    The fit object is updated with the new coefficients, and returned.
    """
    # If no fit object provided, initialize from raw data
    if fit is None:
        fit = UnmatchedFit(x=x, y=y, error_term=error_term, coeffs=coeffs_guess)
    
    # Update the fit
    fit.update()
    
    if len(fit) <= 2:
        raise ValueError("Too few good matches!")
    
    # Now find a better fit for the matches
    coeffs = polyfit(fit.matched_x, fit.matched_y, deg=1)
    fit.update(coeffs)
    
    return fit

def smooth_and_estimate_delay(timestamps1, timestamps2, scales, 
    oversample_ratio=4, p_thresh=.01):
    """Progressively smooth signal and estimate delay.
    
    Smooths signal at each scale in `scales`, then calls `estimate_delay`.
    If a p-value that meets threshold is found, then returns.
    
    TODO: 
        drop part of longer signal if memory quota exceeded
        when a not-quite-good-enough p-value found, then rescale and
        check the same delay at a tighter scale
    
    
    Returns delay_samples, scale:
        delay_samples: number of samples to delay sig2 to achieve good corr
            with sig1
        scale: the scaling that achieved threshold (in seconds/sample)
    
    """
    
    # Find the best delay
    for scale in scales:
        # Put code here to drop from end of longer signal if a memory
        # limit is exceeded
        
        t1, n1, sig1 = PointProc.smooth_at_scale(timestamps1, scale, 
            oversample_ratio=oversample_ratio)
        t2, n2, sig2 = PointProc.smooth_at_scale(timestamps2, scale, 
            oversample_ratio=oversample_ratio)
        
        # Put code here to peek at correlation at previous best delay
        # and do not do full correlation if it is good enough
        
        # estimate the delay
        delay_samples, pval = estimate_delay(sig1, sig2)
        
        # account for the different starting times
        # (which estimate_delay knew nothing about)
        delay_samples = delay_samples + n1[0] -  n2[0]
        delay_sec = delay_samples * scale / oversample_ratio        
        
        if pval < p_thresh:
            break

    if pval > p_thresh:
        print "warning: did not achieve criterion during syncing"

    return delay_samples, delay_sec, scale
    

def estimate_delay(sig1, sig2):
    """Estimates delay between two correlated signals.
    
    Returns: delay, pval
    delay : number of samples to delay sig2 to achieve max correlation
    pval  : Bonferroni corrected p-value of the quality of this correlation,
        relative to the distribution of all other possible delays
        (for whatever that is worth)
    
    TODO: add flag to use scipy.signal.fftconvolve instead
    """
    # Correlate at all lags
    C = np.correlate(sig1, sig2, 'full')
    
    # Now see how good the correlation is
    best = C.max()
    argbest = np.argmax(C) # number of zeros to prepend to smaller sig
    z = (best - np.median(C)) / C.std()
    pval = (1 - ndtr(z))
    pval = 1 - ((1 - pval) ** len(C))
    
    # Subtract off the length of sig1
    res = argbest - len(sig2) + 1
    
    if EVENTSYNC_DEBUG_FIGURE:
        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.add_subplot(121)
        ax.plot(sig1)
        ax.plot(range(res, res + len(sig2)), sig2)
        ax.set_xlim((res, res + len(sig2)))
        
        ax = f.add_subplot(122)
        ax.hist(C, bins=100)
        plt.show()
    
    if EVENTSYNC_DEBUG:
        print ("corr %0.3f at %d (%d) z=%0.2f pval %0.3f" % 
            (best, argbest, res, z, pval))
        sys.stdout.flush()
    
    return res, pval
