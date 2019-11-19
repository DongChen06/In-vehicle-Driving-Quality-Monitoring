Traffic_Project
===============

- Built by Dong Chen, Pengyu Chu, Zhaojian Li from Michigan State University
- Started on Oct.19, 2019, Lastly updated on Oct.22, 2019

Overview
-------

This project aims at building a on-device APP used to asist human drivers. This APP combines three functions: object detection(vehicle, 
traffic light, traffic sign, pedestrain (in future)), lane deviation estimation and distance estimation.

#### Motivation:
To be added...

Part1. Project Building On Ubuntu
-------

### Object Detection Module
To be added...

### Lane Deviation Module

This module is built on the resposity: [Resposity Link](https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding).

Modification logs:
- [x] Delete the display code for "intermediate pipeline images", here we only display the lane deviation and lane curvature.
- [x] Clasify codes.
- [ ] Problems with road curvature and offset values are always positive.

### Distance Estimation Module
47o FOV len.

<p align="center">
     <img src="lane_deviation/Docs/distance_estimation.png" alt="output_example" width="60%" height="60%">
     <br>Distance Estimation
</p>
When camera pitch angle is negligibly small, range d to vehicle can be calculated as in the following: 

```
    d = F_c * H_c / (y_b - y_h)
```

Part2. Project Building On Android
-------

To be added...
