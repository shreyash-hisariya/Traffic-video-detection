# Traffic-video-detection
Detecting main traffic directions and abnormal vehicle movements

The py files contains methods to implement K means algorithms for clustering similar vehicle directions.

The get_features method uses B-spline interpolation technique to build feature set for representing the vehicle movement.

To find outliers, we find the distance between a particular vehicle and the centroid which that vehicle belongs to. If the distance is above a certain threshold, we label it as an outlier.

<img width="785" alt="Screenshot 2021-07-18 at 2 33 42 PM" src="https://user-images.githubusercontent.com/8944710/126078497-f1072be6-5c9e-48b3-841b-5d3825f7de93.png">
