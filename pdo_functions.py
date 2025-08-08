#!/usr/bin/env python
# coding: utf-8

# In[1]:
def detrend_dim(da, dim, deg):
    import xarray as xr
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def convert_longitudes(lon):
    import numpy as np
    lon = np.array(lon)  # Ensure input is a NumPy array
    lon[lon < 0] += 360  # Convert negative longitudes
    return lon


# In[2]:


def preprocess_and_select_months(daily_cape_data, pdo_data, months, anoms):
    ## Data should be a full time series of daily mean or maximum CAPE values
    # Make a climatology from the daily CAPE data
    import xarray as xr
    if months == 'all':
        months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


    monthly_cape = daily_cape_data.resample(time='1M').mean()
    
    if anoms == True:
        climo = monthly_cape.groupby('time.month').mean()
        
        # Remove the climatology to remove the seasonal cycle
        monthly_cape_anoms = monthly_cape.groupby('time.month') - climo

    else:
        monthly_cape_anoms = monthly_cape

    if pdo_data != False:
        # re index time dimension
        monthly_cape_anoms['time'] = pdo_data['time']
        

    # Select months of interest
    # CAPE
    monthly_cape_anoms_select = monthly_cape_anoms.sel(time=monthly_cape_anoms.time.dt.month.isin(months))

    if pdo_data == False:
        return monthly_cape_anoms_select

    else:
        # PDO
        pdo_select = pdo_data.sel(time=pdo_data.time.dt.month.isin(months))
    
        return monthly_cape_anoms_select, pdo_select


# In[3]:


def composite(monthly_anomalies, pdo_data, std_dev, xr_or_np, pos_only):
    import xarray as xr
    import numpy as np

    if xr_or_np == True:
        # First need to broadcast the PDO index to be the same shape as the CAPE array
        pdo_data['time'] = monthly_anomalies['time']
        pdo_broad = pdo_data.broadcast_like(monthly_anomalies)
        pdo_broad = pdo_broad.reindex(time=monthly_anomalies.time)
    else:
        pdo_broad = np.broadcast_to(pdo_data[:, np.newaxis, np.newaxis], np.shape(monthly_anomalies))

    # Find where PDO is positive and negative for the first composite
    pos_pdo_cape = np.nanmean(np.where(pdo_broad > std_dev, monthly_anomalies, np.nan), axis=0)
    neg_pdo_cape = np.nanmean(np.where(pdo_broad < (-1*std_dev), monthly_anomalies, np.nan), axis=0)

    # Number of months of each
    n_pos = len(pdo_data[pdo_data > std_dev])
    n_neg = len(pdo_data[pdo_data < (-1*std_dev)])

    if pos_only == False:
        return pos_pdo_cape, neg_pdo_cape, n_pos, n_neg
    else:
        return pos_pdo_cape, n_pos


# In[ ]:

def oni_moving_base(sst_data):
    import xarray as xr
    import numpy as np
    ## Assumes that the time period of analysis is 1870-2024
    ## Returns the monthly ONI index from 1885-2024
    
    # Subset nino_34 region
    nino_34 = sst_data.sel(longitude=slice(-170, -120)).sel(latitude=slice(5, -5))

    # Take the weighted area average
    weights = np.cos(np.deg2rad(nino_34.latitude))

    nino_34_mean = nino_34.weighted(weights).mean(dim=('latitude', 'longitude'))

    # Define base period and test period start and end years
    base_period_start_years = np.arange(1871, 1996, 5)
    base_period_end_years = np.arange(1900, 2021, 5)

    test_period_start_years = np.arange(1886, 2011, 5)
    test_period_end_years = np.arange(1890, 2014, 5)

    # Compute monthly SST anoms
    sst_anoms = np.zeros((1668))
    for i in range(26):
        if i < 25:
            # Compute SST climo 
            climo = nino_34_mean.sel(time=nino_34_mean.time.dt.year.isin(np.arange(base_period_start_years[i], base_period_end_years[i]+1, 1))).groupby('time.month').mean()
        
            # Subset years of interest
            yrs_of_int = nino_34_mean.sel(time=nino_34_mean.time.dt.year.isin(np.arange(test_period_start_years[i], test_period_end_years[i]+1, 1)))
        
            # Compute anomalies and store
            sst_anoms[i*60:(i+1)*60] = yrs_of_int.groupby('time.month') - climo
            
            # Prove that you used the right base period and test period
            # print(f'base period: {base_period_start_years[i]}-{base_period_end_years[i]}, test period: {test_period_start_years[i]}-{test_period_end_years[i]}')
        else: 
            # Same climo as before (1991-2020)
            climo = nino_34_mean.sel(time=nino_34_mean.time.dt.year.isin(np.arange(base_period_start_years[24], base_period_end_years[24]+1, 1))).groupby('time.month').mean()
    
            # Years of interest
            yrs_of_int = nino_34_mean.sel(time=nino_34_mean.time.dt.year.isin(np.arange(2011, 2025, 1)))
    
            # Compute anomalies and store
            sst_anoms[1500:] = yrs_of_int.groupby('time.month') - climo

    # Convert sst_anoms back to XR
    sst_anoms_xr = xr.DataArray(sst_anoms, coords={'time':sst_data['time'][12*16:]}, dims=['time'])

    return sst_anoms_xr

    

def calc_nino_34_timeseries(sst_data, lon_360_or_lon_180, first_base_year, second_base_year):
    import numpy as np
    import xarray as xr
    # Subset nino3.4 region
    if lon_360_or_lon_180 == True:
        sst_nino_34 = sst_data.sel(longitude=slice(pdo_functions.convert_longitudes(-170), pdo_functions.convert_longitudes(-120))).sel(latitude=slice(5, -5))
    elif lon_360_or_lon_180 == False:
        sst_nino_34 = sst_data.sel(longitude=slice(-170, -120)).sel(latitude=slice(5, -5))

    # Area weights
    sst_nino_34_weights = np.cos(np.deg2rad(sst_nino_34.latitude))

    # Apply weights and take area average
    sst_nino_34_mean = sst_nino_34.weighted(sst_nino_34_weights).mean(dim=('latitude', 'longitude'))

    # Make a monthly climo w.r.t 1981-2010
    sst_nino_34_climo = sst_nino_34_mean.sel(time=sst_nino_34_mean.time.dt.year.isin(np.arange(first_base_year, second_base_year+1, 1))).groupby('time.month').mean()

    # Make monthly anomalies timeseries
    sst_nino_34_anoms = sst_nino_34_mean.groupby('time.month') - sst_nino_34_climo

    return sst_nino_34_anoms



def pdo_from_hadisst(sst_data, first_base_year, second_base_year):
    ## CALCULATE THE PDO ACCORDING TO THE NOAA PSL METHODS (see: https://psl.noaa.gov/pdo/)
    ## Note: there is some bias, especiialy in the recent decade or so,... this may be because this method
    ##       does not regrid to a 2x2degree resolution
    
    import xarray as xr
    import numpy as np
    import regionmask
    from geocat.comp import eofunc_eofs, eofunc_pcs

    # land mask
    land_110 = regionmask.defined_regions.natural_earth_v4_1_0.land_110
    land = xr.where(land_110.mask_3D(sst_data['longitude'], sst_data['latitude'])==False, 1, np.nan).squeeze()
    
    # SST with no land
    sst_no_land = sst_data*land
    
    # Fill really negative values with a constant value
    sst_no_land_fill = sst_no_land.where(sst_no_land > -1.8, -1.8)

    # Trying out 1960-2020 base period
    sst_no_land_fill_1920_2014 = sst_no_land_fill.sel(time=sst_no_land_fill.time.dt.year.isin(np.arange(first_base_year, second_base_year+1, 1)))
    
    # Create monthly climo and remove seasonal cycle
    sst_no_land_fill_1920_2014_climo = sst_no_land_fill_1920_2014.groupby('time.month').mean()
    
    sst_no_land_fill_no_climo = sst_no_land_fill.groupby('time.month') - sst_no_land_fill_1920_2014_climo
    
    # Select only 1920-2014 - testing out calculating the PDO index using the whole time series
    # sst_no_land_fill_no_climo_1920_2014 = sst_no_land_fill_no_climo.sel(time=sst_no_land_fill_no_climo.time.dt.year.isin(np.arange(first_base_year, second_base_year+1, 1)))
    
    # Calculate the global mean SST for each month and remove, for some reason omitting polar latitudes?
    sst_global_mean = sst_no_land_fill_no_climo.sel(latitude=slice(70, -60)).weighted(np.cos(np.deg2rad(sst_no_land_fill.sel(latitude=slice(70, -60)).latitude))).mean(dim=('latitude', 'longitude'))
    # sst_global_mean = sst_no_land_fill_no_climo.weighted(np.cos(np.deg2rad(sst_no_land_fill.latitude))).mean(dim=('latitude', 'longitude'))
    
    sst_no_land_fill_anoms = sst_no_land_fill_no_climo - sst_global_mean

    # sst_no_land_fill_anoms_1920_2014 = sst_no_land_fill_anoms.sel(time=sst_no_land_fill_anoms.time.dt.year.isin(np.arange(first_base_year, second_base_year+1, 1)))
    
    # EOF analysis
    # Regional subset
    sst_no_land_fill_anoms_pdo = xr.concat([sst_no_land_fill_anoms.sel(latitude=slice(70, 20)).sel(longitude=slice(110, 180)), \
                                            sst_no_land_fill_anoms.sel(latitude=slice(70, 20)).sel(longitude=slice(-180, -100))], dim='longitude')
    
    # Weight by the sqrt of the cosine of the latitude
    sst_no_land_fill_anoms_pdo_wgt = sst_no_land_fill_anoms_pdo * np.sqrt(np.cos(np.deg2rad(sst_no_land_fill_anoms_pdo.latitude)))
    
    # Subset 1920-2014
    # sst_no_land_fill_anoms_pdo_wgt_1920_2014 = sst_no_land_fill_anoms_pdo_wgt.sel(time=sst_no_land_fill_anoms_pdo_wgt.time.dt.year.isin(np.arange(first_base_year, second_base_year+1, 1)))
    
    # EOF
    eof = eofunc_eofs(sst_no_land_fill_anoms_pdo_wgt, neofs=1)
    
    eof_reshape = np.reshape((eof[0].values*-1), 50*150)
    
    ssta_tseries_reshape = np.reshape(sst_no_land_fill_anoms_pdo_wgt.transpose('latitude', 'longitude', 'time').values, (50*150, 1860))
    
    pcs = eofunc_pcs(sst_no_land_fill_anoms_pdo.data, npcs=1)
    
    # standardize the PC
    standard_pcs = (pcs[0] - pcs[0].mean()) / pcs[0].std()
    
    # define my own dot product function for when there are NaNs.
    # Note that this does the scaling by 1/M inside it (by taking nanmean), so not a true dot product.
    def nandot(X, Y):
    
        C = np.empty([np.size(Y, axis=1)])
        for time in np.arange(0, np.size(Y, axis=1)):
            C[time] = np.nanmean(np.multiply(X, Y[:, time]))
    
        return C
    
    pcs_full = nandot(eof_reshape, ssta_tseries_reshape)
    
    pcs_full_std = (pcs_full - np.mean(pcs_full)) / np.std(pcs_full)

    # return (pcs_full_std)
    return standard_pcs


def count_in_grid(sorted_storm_report_lat, sorted_storm_report_lon, sorted_grid_data_lat, sorted_grid_data_lon):
    import numpy as np
    lat_bins = sorted_grid_data_lat
    lon_bins = sorted_grid_data_lon

    hist, _, _ = np.histogram2d(sorted_storm_report_lat, sorted_storm_report_lon, bins=[lat_bins, lon_bins])

    return hist

import numba
import numpy as np
@numba.jit(nopython=True, parallel=True)
def bootstrap_p_values(n_lats, n_lons, sample_mean, cape_climo, comp_cond):
    ## n_lats should be an integer of the number of latitude grid points
    ## n_lons should be an integer of the number of longitude grid points
    ## sample mean should be the time mean of the sample (with dimensions of (n_lat, n_lon)
    ## cape_climo should be the monthly mean de-seasonalized CAPE anomalies with shape (n_lat, n_lon, months)
    ## comp_cond should utilize the PDO (or related index) monthly time series with the same number of months as 
    ## cape_climo and condition on what the desired threshold is (i.e., +1 standard deviations)

    ## This function creates 10000 bootstrapped samples to identify if composites of different climate modes
    ## (such as the Pacific Decadal Oscillation) are statistically significantly different than the climatology
    ## return is p-values at each grid point
    p_values = np.zeros((n_lats, n_lons))
    for lat in range(n_lats):
        for lon in range(n_lons):
            # Sample mean
            test_mean = sample_mean[lat][lon]
    
            # Select lat and lon point from the monthly CAPE anomaly dataset
            climo_point = cape_climo[lat][lon]
    
            # Now subset according to what composite you are testing the significance of
            climo_point_comp = climo_point[comp_cond]
    
            # Create 10000 bootstrapped samples
            boot_means = np.zeros((10000))
            for i in range(10000):
                boot_sample = np.zeros((len(climo_point_comp)))
                for j in range(len(climo_point_comp)):
                    idx = np.random.randint(0, len(climo_point))
                    boot_sample[j] = climo_point[idx]
    
                boot_means[i] = np.mean(boot_sample)

            climo_mean = np.mean(climo_point)
            # Calculate the p-values
            p_val = np.mean(np.abs(boot_means-climo_mean) >= np.abs(test_mean-climo_mean))
    
            p_values[lat][lon] = p_val

        print(f'{lat} is complete')

    return p_values


def control_FDR(p_values, n_lat, n_lon, alpha):
    import numpy as np

    # Employing consideration of the False Discovery Rate
    # Using methods described in Wilks et al. (2016)
    # Step 1: Reshape and sort p-values
    p_val_sorted = np.sort(np.reshape(p_values, n_lat*n_lon))
    
    # Step 2: Create an array of the same size as that in (1)
    index = np.arange(0, len(p_val_sorted))
    
    # Step 3: Create function to compare to sorted p-values
    y = alpha*(index/len(index))
    
    # Step 4: Find where the sorted p-values crosses y
    crossing = np.where(y >= p_val_sorted)
    
    # Step 5: identify the 'adjusted p-value' by identifying the index of interest
    #         and then multiply by 0.5 to account for spatial and temporal autocorrelation

    if np.size(crossing) > 0:
        new_p = p_val_sorted[crossing[0][-1]]

        return new_p
    else:
        return 0

def linregress_3D(x, y):
    import warnings
    """
    Performs linear regression between two data arrays along the time dimension.

    Parameters:
    x (xarray.DataArray): Independent variable with the first dimension being time. Can be multi-dimensional.
    y (xarray.DataArray): Dependent variable with the first dimension being time. Can be multi-dimensional.

    Returns:
    tuple: Covariance, correlation, regression slope, and intercept arrays.

    Adapted from: http://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    """
    # Suppress warnings that may arise from operations involving NaNs or other runtime issues
    # This ensures that warnings do not interrupt the function execution and clutter the output

    ## NOTE that time arrays have to be aligned exactly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Compute data length, mean, and standard deviation along the time axis for further use
        n = np.shape(x)[0]

        if x.ndim == 1:
            x = x[:, np.newaxis, np.newaxis]
        
        xmean = np.nanmean(x, axis=0)
        ymean = np.nanmean(y, axis=0)
        xstd = np.nanstd(x, axis=0)
        ystd = np.nanstd(y, axis=0)

        # Compute covariance along the time axis
        cov = np.nansum((x - xmean) * (y - ymean), axis=0) / n

        # Compute correlation along the time axis
        cor = cov / (xstd * ystd)

        # Compute regression slope and intercept
        slope = cov / (xstd ** 2)
        intercept = ymean - xmean * slope

    return cov, cor, slope, intercept