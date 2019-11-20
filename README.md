Traffic_Project
===============

- Built by Dong Chen, Pengyu Chu, Zhaojian Li from Michigan State University
- Started on Oct.19, 2019, Lastly updated on Nov.19, 2019

Overview
-------

This project aims at building a on-device APP used to asist human drivers. This APP combines three basic functions: object detection(vehicle, 
traffic light, traffic sign, pedestrain), lane deviation warning and distance estimation.

#### Motivation:
To be added...

Part1. Project Built Offline
-------

### Object Detection Module
We use the deep learning methods to do object detection. To be specific, we use the [YOLO-v3]( https://pjreddie.com/darknet/yolo/) model to do object detection, here we are only curious about traffic-related objects, such as vehicles, pedestrain, traffic lights and stop signs.
This module is built on the resposity: [Resposity Link](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch).

### Lane Deviation Module
Considered limited computing resources on mobile devices (smart phones), we adapt the convential computer vision methods. 

This module is built on the resposity: [Resposity Link](https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding).

Modification logs:
- [x] Delete the display code for "intermediate pipeline images".
- [x] Simiplify codes.
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

### Demos:
For privacy issues, there are few open resources for dash camera videos. We will show our application by three different video demos.
- Stop sign detection

     [video link](https://www.youtube.com/watch?v=jctnZcWk2uQ)

- Traffic light detection

     [video link](https://www.youtube.com/watch?v=7ZbgyEO9cro)

- Lane deviation detection

     [video link](https://www.youtube.com/watch?v=kqG6j8lDm84)


Part2. Project Built On Android
-------

To be added...
