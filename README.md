<!-- * if you have time, update this readme file for that it is outdated (last updated in 2021.10) -->

# BITFSD Trajectory Optimization Program

## Motivation 

The curvature of cubic spline is **zigzagging**, as is shown in the following figure. When sampling discretely, this leads to unexpected peaks and troughs that deviate sampled curve from the real one. 

<p align="center"> 
<img src="doc/image/cubic spline curvature.svg" width="60%">
</p>

This program aims to smooth the curvature while interpolating fixed center points of track. And further perform global optimization at the limit of driving condition. A link to more detailed description of optimization method will be provided in the future. 

Comparisons between cubic spline and optimized curve and curvature are shown below.

<p align="center"> 
<img src="doc/image/curvature optimization results.svg">
</p>

<!-- Based on the reference curve with optimzied curvature, further efforts are paid to generate a dynamically optimal curve.  -->

## How to Use

### Install opt

`opt` is the core package in this project, storing major algorithms. `poetry` is recommended to manage dependencies and packaging. Install `poetry` through `pip3 install poetry` and run `poetry install`. This command installs both dependencies and the prescibed package of this project to the virtual enviroment.

### Demo

You can run the demo file for a test for curvature optimization first. It will generate an image simialr to the one above and print optimization details. 

```sh
python3 demo.py
```

### APP

To use the application, copy the `map.txt` file to the `BITFSD_trajectory_optimization/data` directory and run:

```sh
python3 app.py
```

The file `?_?.yaml` containing `2-D coordinates` & `kappa (curvatures)` & `theta (yaw angles)` & `left right track boundaries` of discrete center points will be generated under `BITFSD_trajectory_optimization/data`. The two `?`s stands for optimization types.

## Parameters

written in `BITFSD_trajectory_optimization/config/config.yaml`.

***Centerline Optimization***

- `enable_plot`: plotting is usually needed to inspect the curve state. Curvature should be within [-0.8, 0.8]. 

- `interval`: approximate distance between optimized discrete waypoints.

- `a`, `b`, `g`: coefficients in objective function related to first derivative, second derivative of curvature and curvature itself.

## Requirements

The `map.txt` file should have at least one empty line at the end, or the last line will be ommited. 

## Dependencies

This program runs on ***python3***. More specific dependencies can be found in `pyproject.toml`.

## Lisense

> A copy of this program will not be lisensed before 2022.08.30.

Copyright © 2021, [Zijun Guo](https://github.com/Easy121). All Rights Reserved.