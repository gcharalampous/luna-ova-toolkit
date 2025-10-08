"""
OFDR Analysis Utilities

This module provides utility functions for analyzing OFDR (Optical Frequency Domain Reflectometry) data.

Author: George Charalambous
Date: 2025
License: MIT
"""

import pandas as pd
from typing import Literal
from io import StringIO
import os
from pathlib import Path


def read_ofdr_txt(path):
    """
    Reads an OFDR reflection file exported as a .txt (Length vs Amplitude).
    Automatically skips the metadata header and loads numeric data into a DataFrame.
    
    Parameters
    ----------
    path : str or Path
        Path to the OFDR .txt file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Length_m' and 'Amplitude_dB'
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If no valid numeric data is found in the file
    UnicodeDecodeError
        If the file cannot be decoded (handled gracefully with 'ignore')
        
    Examples
    --------
    >>> df = read_ofdr_txt("sample_ofdr_data.txt")
    >>> print(df.columns)
    Index(['Length_m', 'Amplitude_dB'], dtype='object')
    >>> print(f"Loaded {len(df)} data points")
    """
    path = Path(path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"OFDR file not found: {path}")
    
    # Read all lines and find where the numeric table starts
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}")

    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Length") and "Amplitude" in line:
            start_idx = i + 1
            break

    # Convert the numeric block into a Pandas DataFrame
    numeric_block = "".join(lines[start_idx:])
    df = pd.read_csv(
        StringIO(numeric_block),
        sep=r'\s+',                 # columns are separated by whitespace
        header=None,                # no explicit header row in numeric block
        names=["Length_m", "Amplitude_dB"]  # assign column names
    )

    # Clean non-numeric values (if any)
    df = df[
        pd.to_numeric(df["Length_m"], errors="coerce").notna() &
        pd.to_numeric(df["Amplitude_dB"], errors="coerce").notna()
    ]
    
    # Convert to proper numeric types
    df["Length_m"] = pd.to_numeric(df["Length_m"])
    df["Amplitude_dB"] = pd.to_numeric(df["Amplitude_dB"])
    
    # Validate that we have data
    if len(df) == 0:
        raise ValueError(f"No valid numeric data found in file: {path}")
    
    return df


def get_file_info(path):
    """
    Get basic information about an OFDR data file.
    
    Parameters
    ----------
    path : str or Path
        Path to the OFDR .txt file
        
    Returns
    -------
    dict
        Dictionary containing file information:
        - 'filename': Name of the file
        - 'size_bytes': File size in bytes
        - 'data_points': Number of data points (if readable)
        - 'length_range': Tuple of (min_length, max_length) in meters
        - 'amplitude_range': Tuple of (min_amplitude, max_amplitude) in dB
        
    Examples
    --------
    >>> info = get_file_info("sample_ofdr_data.txt")
    >>> print(f"File has {info['data_points']} data points")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"OFDR file not found: {path}")
    
    info = {
        'filename': path.name,
        'size_bytes': path.stat().st_size,
        'data_points': None,
        'length_range': None,
        'amplitude_range': None,
    }
    
    try:
        df = read_ofdr_txt(path)
        info['data_points'] = len(df)
        info['length_range'] = (df['Length_m'].min(), df['Length_m'].max())
        info['amplitude_range'] = (df['Amplitude_dB'].min(), df['Amplitude_dB'].max())
    except Exception:
        # If we can't read the data, just return basic file info
        pass
    
    return info

def convert_amplitude_to_linear(df):
    """
    Convert amplitude from dB to linear scale.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Amplitude_dB' column
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional 'Amplitude_linear' column
    """
    df = df.copy()
    df['Amplitude_linear'] = 10 ** (df['Amplitude_dB'] / 10)
    return df


def find_reflection_peaks(df, threshold=-70, distance=None):
    
    """
    Identify reflection peaks in OFDR data using scipy's find_peaks.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Length_m' and 'Amplitude_dB' columns
    threshold : float, optional
        Minimum amplitude (in dB) to consider a peak significant (default is -70 dB)
    distance : int, optional
        Minimum number of samples between adjacent peaks (default is None)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with identified peaks, including 'Length_m', 'Amplitude_dB', and peak properties
        
    Examples
    --------
    >>> peaks_df = find_reflection_peaks(df, threshold=-70, distance=10)
    >>> print(peaks_df)
    """
    from scipy.signal import find_peaks
    
    # Ensure input DataFrame has required columns
    if 'Length_m' not in df.columns or 'Amplitude_dB' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Length_m' and 'Amplitude_dB' columns")
    
    # Find peaks using scipy's find_peaks
    peaks, properties = find_peaks(df['Amplitude_dB'], height=threshold, distance=distance)
    
    # Create a DataFrame for the peaks
    peaks_df = pd.DataFrame({
        'Length_m': df['Length_m'].iloc[peaks].values,
        'Amplitude_dB': df['Amplitude_dB'].iloc[peaks].values,
        **{key: properties[key] for key in properties}
    })
    
    return peaks_df
    
    
def filter_data_by_length(df, min_length=None, max_length=None):
    """
    Filter OFDR data by length range.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Length_m' column
    min_length : float, optional
        Minimum length (in meters) to include (default is None)
    max_length : float, optional
        Maximum length (in meters) to include (default is None)
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame
        
    Examples
    --------
    >>> filtered_df = filter_data_by_length(df, min_length=0.1, max_length=1.0)
    >>> print(filtered_df)
    """
    if min_length is not None:
        df = df[df['Length_m'] >= min_length]
    if max_length is not None:
        df = df[df['Length_m'] <= max_length]
    return df    


def linear_fit_within_range(df, min_length, max_length):
    """
    Perform a linear fit on OFDR data within a specified length range.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Length_m' and 'Amplitude_dB' columns
    min_length : float
        Minimum length (in meters) for fitting
    max_length : float
        Maximum length (in meters) for fitting
        
    Returns
    -------
    tuple
        (slope, intercept) of the linear fit in dB/m
        
    Examples
    --------
    >>> slope, intercept = linear_fit_within_range(df, 0.1, 1.0)
    >>> print(f"Slope: {slope} dB/m, Intercept: {intercept} dB")
    """
    from scipy.stats import linregress
    
    # Filter data within the specified length range
    fit_df = df[(df['Length_m'] >= min_length) & (df['Length_m'] <= max_length)]
    
    if len(fit_df) < 2:
        raise ValueError("Not enough data points within the specified length range for fitting.")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(fit_df['Length_m'], fit_df['Amplitude_dB'])
    
    # Plot the fitted line along the data
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(df['Length_m'], df['Amplitude_dB'], label='OFDR Data', color='blue')
    plt.plot(
        fit_df['Length_m'],
        slope * fit_df['Length_m'] + intercept,
        color='red',
        linestyle='--',
        linewidth=3,
        label='Linear Fit'
    )
    plt.xlabel('Length (m)')
    plt.ylabel('Amplitude (dB)')
    plt.title('OFDR Data with Linear Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return slope, intercept

def plot_ofdr_data(df, peaks_df=None):
    """
    Plot OFDR data with optional reflection peaks highlighted.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Length_m' and 'Amplitude_dB' columns
    peaks_df : pandas.DataFrame, optional
        DataFrame with identified peaks to highlight (same format as output of find_reflection_peaks)
        
    Returns
    -------
    None
        Displays a plot of the OFDR data
        
    Examples
    --------
    >>> plot_ofdr_data(df, peaks_df)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Length_m'], df['Amplitude_dB'], label='OFDR Data', color='blue')
    
    if peaks_df is not None and not peaks_df.empty:
        plt.scatter(peaks_df['Length_m'], peaks_df['Amplitude_dB'], color='red', label='Reflection Peaks', zorder=5)
    
    plt.xlabel('Length (m)')
    plt.ylabel('Amplitude (dB)')
    plt.title('OFDR Reflection Data')
    plt.legend()
    plt.grid(True)
    plt.show()