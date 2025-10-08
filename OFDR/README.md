# OFDR Analysis for Silicon Spiral Waveguides

This repository contains Python tools for analyzing Optical Frequency Domain Reflectometry (OFDR) data from silicon spiral waveguides. The project focuses on calibration measurements and provides utilities for processing OFDR reflection data.

## Overview

OFDR (Optical Frequency Domain Reflectometry) is a technique used to measure the distributed characteristics of optical devices. This project provides tools to:

- Load and parse OFDR data files exported as text format
- Analyze reflection amplitude vs. length measurements
- Compare different waveguide geometries and configurations
- Process calibration data for silicon spiral waveguides

## Features

- **Automated Data Loading**: Parse OFDR text files with automatic header detection
- **Clean Data Processing**: Remove invalid data points and handle formatting issues
- **Jupyter Notebook Analysis**: Interactive analysis environment
- **Modular Design**: Reusable utility functions for OFDR data processing

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

# Luna OVA — OFDR Data Processing

Utility scripts and notebooks for processing OFDR measurement outputs from Luna OVA systems.

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the demo notebook:

```bash
jupyter notebook ofdr_analysis.ipynb
```


License: MIT