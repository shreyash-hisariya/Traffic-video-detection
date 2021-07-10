# Traffic-video-detection
Detecting main traffic directions and abnormal vehicle movements

The py files contains methods to implement K means algorithms for clustering similar vehicle directions.

The get_features method uses B-spline interpolation technique to build feature set for representing the vehicle movement.

To find outliers, we find the distance between a particular vehicle and the centroid which that vehicle belongs to. If the distance is above a certain threshold, we label it as an outlier.
