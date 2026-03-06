# These routines are largely copied from the E3SM diagnostics package in order
# to cacluate identical metrics for QBO quantities of interest
#---------------------------------------------------------------------------------------------------
import numpy as np, xarray as xr, scipy, numba
#---------------------------------------------------------------------------------------------------
# @numba.jit(nopython=True)
def calculate_area_from_latlon(lon,lat,lon_bnds,lat_bnds):
    re = 6.37122e06  # radius of earth
    nlat,nlon = len(lat),len(lon)
    area = np.empty((nlat,nlon),np.float64)
    for j in range(nlat):
        for i in range(nlon):
            dlon = np.absolute( lon_bnds[j,1] - lon_bnds[j,0] )
            dlat = np.absolute( lat_bnds[j,1] - lat_bnds[j,0] )
            dx = re*dlon*np.pi/180.
            dy = re*dlat*np.pi/180.
            area[j,i] = dx*dy
    return area
#---------------------------------------------------------------------------------------------------
# def calculate_obs_area(lon,lat,lon_bnds,lat_bnds):
#    re = 6.37122e06  # radius of earth
#    nlat,nlon = len(lat),len(lon)
#    area = np.empty((nlat,nlon),np.float64)
#    for j in range(nlat):
#       for i in range(nlon):
#          dlon = np.absolute( lon_bnds[j,1] - lon_bnds[j,0] )
#          dlat = np.absolute( lat_bnds[j,1] - lat_bnds[j,0] )
#          dx = re*dlon*np.pi/180.
#          dy = re*dlat*np.pi/180.
#          area[j,i] = dx*dy
#    return area
#---------------------------------------------------------------------------------------------------
def deseason(xraw):
    # Calculates the deseasonalized data
    months_per_year = 12
    # Create array to hold climatological values and deseasonalized data
    # Create months_per_year x 1 array of zeros
    xclim = np.zeros((months_per_year, 1))
    # Create array with same shape as xraw
    x_deseasoned = np.zeros(xraw.shape)
    # Iterate through all 12 months.
    for month in np.arange(months_per_year):
        # `xraw[month::12]` will return the data for this month every year (12 months)
        # (i.e., from month until the end of xraw, get every 12th month)
        # Get the mean of this month, using data from every year, ignoring NaNs
        xclim[month] = np.nanmean(xraw[month::months_per_year])
    num_years = int(np.floor(len(x_deseasoned) / months_per_year))
    # Iterate through all years in x_deseasoned (same number as in xraw)
    for year in np.arange(num_years):
        year_index = year * months_per_year
        # Iterate through all months of the year
        for month in np.arange(months_per_year):
            month_index = year_index + month
            # Subtract the month's mean over num_years from xraw's data for this month in this year
            # i.e., get the difference between this month's value and it's "usual" value
            x_deseasoned[month_index] = xraw[month_index] - xclim[month]
    return x_deseasoned
#---------------------------------------------------------------------------------------------------
def ceil_log2(x):
    """
    Given a number, calculate the exponent for the next power of 2.
    Example:
        ceil_log2(16) = 4
        ceil_log2(17) = 5
    """
    return np.ceil(np.log2(x)).astype("int")
#---------------------------------------------------------------------------------------------------
def get_psd_from_deseason(xraw, period_new):
    x_deseasoned = deseason(xraw)

    # Sampling frequency: assumes frequency of sampling = 1 month
    sampling_frequency = 1
    # Calculate the period as a function of frequency
    period0 = 1 / sampling_frequency
    L0 = len(xraw)
    NFFT0 = 2 ** ceil_log2(L0)

    # Apply fft on x_deseasoned with n = NFFT
    x0 = scipy.fftpack.fft(x_deseasoned, n=NFFT0) / L0
    # Frequency (cycles/month). Frequency will be increasing.
    frequency0 = sampling_frequency * np.arange(0, (NFFT0 / 2 + 1)) / NFFT0
    # Period (months/cycle). Calculate as a function of frequency. Thus, period will be decreasing.
    period0 = np.zeros_like(frequency0)
    for f,freq in enumerate(frequency0):
        period0[f] = 0 if freq==0 else 1/freq

    # Calculate amplitude as a function of frequency
    amplitude0 = 2 * abs(x0[0 : int(NFFT0 / 2 + 1)])
    # Calculate power spectral density as a function of frequency
    psd_x0 = amplitude0**2 / L0
    # Total spectral power
    # In the next code block, we will perform an interpolation using the period
    # (interpolating values of amplitude0_flipped and psd_x0_flipped from period0_flipped to period_new).
    # For that interpolation, we want the period to be increasing.
    # Therefore, we will flip the following values:
    period0_flipped = period0[::-1]  # type: ignore
    amplitude0_flipped = amplitude0[::-1]
    psd_x0_flipped = psd_x0[::-1]

    amplitude_new0 = np.interp(period_new, period0_flipped[:-1], amplitude0_flipped[:-1])
    psd_x_new0 = np.interp(period_new, period0_flipped[:-1], psd_x0_flipped[:-1])
    return psd_x_new0, amplitude_new0
#---------------------------------------------------------------------------------------------------
def get_20to40month_fft_amplitude(qboN, levelN):
    # Calculates the amplitude of wind variations in the 20 - 40 month period
    psd_sumN = np.zeros(levelN.shape,dtype="complex_")
    amplitudeN = np.zeros(levelN.shape)

    for ilev in np.arange(len(levelN)):
        # `qboN[:, ilev]` returns the entire 0th dimension for ilev in the 1st dimension of the array.
        y_input = deseason(np.squeeze(qboN[:, ilev]))
        y = scipy.fftpack.fft(y_input)
        n = len(y)
        frequency = np.arange(n / 2) / n
        period = np.full(len(frequency),np.inf)
        period[1:] = 1 / frequency[1:]
        values = y[0 : int(np.floor(n / 2))]
        fyy = values * np.conj(values)
        # Choose the range 20 - 40 months that captures most QBOs (in nature)
        psd_sumN[ilev] = 2 * np.nansum(fyy[(period <= 40) & (period >= 20)])
        amplitudeN[ilev] = np.real( np.sqrt(2 * psd_sumN[ilev]) * (frequency[1] - frequency[0]) )
    return psd_sumN, amplitudeN
#---------------------------------------------------------------------------------------------------
