"""

From wavelets.py
https://pypi.python.org/pypi/agilegeo/
"""
import numpy as np
from scipy.signal import hilbert
from scipy.signal import chirp


def ricker(duration, dt, f):
    """
    Also known as the mexican hat wavelet, models the function:
    A =  (1-2 \pi^2 f^2 t^2) e^{-\pi^2 f^2 t^2}

    :param duration: The length in seconds of the wavelet.
    :param dt: is the sample interval in seconds (usually 0.001,
               0.002, 0.004)
    :params f: Center frequency of the wavelet (in Hz). If a list or tuple is
               passed, the first element will be used.

    :returns: ricker wavelets with center frequency f sampled at t.
    """


    freq = np.array(f)
     
    t = np.arange(-duration/2, duration/2 , dt)

    output = np.zeros((t.size, freq.size))
        
    for i in range(freq.size):
        pi2 = (np.pi ** 2.0)
        if ( freq.size == 1 ):
            fsqr = freq ** 2.0
        else:
            fsqr =  freq[i] ** 2.0
        tsqr = t ** 2.0
        pft = pi2 * fsqr * tsqr
        A = (1 - (2 * pft)) * np.exp(-pft)
        output[:,i] = A

    if freq.size == 1: output = output.flatten()

    A = output / np.amax(output)
        
    return t, A
    
    
def sweep(duration, dt, f, method = 'linear', phi = 0,
          vertex_zero = True, autocorrelate = True):
    """
    Generates a linear frequency modulated wavelet (sweep)
    Does a wrapping of scipy.signal.chirp
    
    :param duration: The length in seconds of the wavelet.
    :param dt: is the sample interval in seconds (usually 0.001, 0.002, 0.004)
    :param f: Tuple of (f1, f2), or a similar list. A list of lists
              will create a wavelet bank.
    :keyword method: {'linear','quadratic','logarithmic'}, optional
    :keyword phi: float, phase offset in degrees
    :keyword vertex_zero: bool, optional
        This parameter is only used when method is 'quadratic'. 
        It determines whether the vertex of the parabola that 
        is the graph of the frequency is at t=0 or t=t1.

    :returns: An LFM waveform.
    """
    
    t = np.arange(-duration/2, duration/2 , dt) 
    t0 = -duration/2
    t1 = duration/2

    freq = np.array( f )

    if freq.size == 2:
         A = chirp(t, freq[0], t1, freq[1],
                    method, phi, vertex_zero)
         if autocorrelate:
             A = np.correlate(A, A, mode='same')
         output =  A / np.amax(A)
    
    else:
        output = np.zeros((t.size, freq.shape[1]))
        
        for i in range( freq.shape[1] ):
            A = chirp(t,freq[0,i],t1,freq[1,i],
                       method, phi, vertex_zero)
    
            if autocorrelate:
                A = np.correlate(A, A, mode='same')
            output[:,i] = A / np.max(A)
     
    return t, output


def ormsby(duration, dt, f):
    """
    The Ormsby wavelet requires four frequencies:
    f1 = low-cut frequency
    f2 = low-pass frequency
    f3 = high-pass frequency
    f4 = hi-cut frequency
    Together, the frequencies define a trapezoid shape in the
    spectrum.
    The Ormsby wavelet has several sidelobes, unlike Ricker wavelets
    which only have two, one either side.

    :param duration: The length in seconds of the wavelet.
    :param dt: is the sample interval in seconds (usually 0.001,
               0.002, 0.004)
    :params f: Tuple of form (f1,f2,f3,f4), or a similar list.

    :returns: A vector containing the ormsby wavelet
    """
    
    # Try to handle some duck typing
    if not (isinstance(f, list) or isinstance(f, tuple)):
        f = [f]
    
    # Deal with having fewer than 4 frequencies
    if len(f) == 4:
        f1 = f[0]
        f2 = f[1]
        f3 = f[2]
        f4 = f[3]
    else:
        # Cope with only having one frequency
        # This is an arbitrary hack, is this desirable?
        # Need a way to notify with warnings
        f1 = f[0]/4
        f2 = f[0]/2
        f3 = f[0]*2
        f4 = f[0]*2.5
    
    def numerator(f,t):
        return (np.sinc( f * t)**2) * ((np.pi * f) ** 2)

    pf43 = ( np.pi * f4 ) - ( np.pi * f3 )
    pf21 = ( np.pi * f2 ) - ( np.pi * f1 )
    
    t = np.arange(-duration/2, duration/2 , dt) 
    
    A = ((numerator(f4,t)/pf43) - (numerator(f3,t)/pf43) -
         (numerator(f2,t)/pf21) + (numerator(f1,t)/pf21))

    A /= np.amax(A)
    return t, A


def rotate_phase(w, phi):
    """
    Performs a phase rotation of wavelet using:
    A = w(t)Cos(phi) + h(t)Sin(phi)
    Where w(t) is the wavelet and h(t) is it's hilbert transform.
    
    :params w: The wavelet vector.
    :params phi: The phase rotation angle (in Radians) to apply.

    :returns: The phase rotated signal.
    """

    # Get the analytic signal for the wavelet
    a = hilbert(w, axis=0)

    A = (np.real(a) * np.cos(phi) +
         np.imag(a) * np.sin(phi))

    return A
    
