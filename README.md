# LHL-Final-Project-Natural-Downtime

### Project Goal

Find the natural downtime of a fleet manegement vehicle in order to optimally provide onsite maintenance servicing. The natural downtime of a vehicle is the location where the vehicle most frequently stops for the longest durations of time (ex. The home of the person who owns the vehicle or the headquarters of the fleet management company, etc.)

### Dataset

The dataset is contrived from tracking the trips and location of 1 vehicle over the span of 1 month. The data had to be joined between 2 datasets, one consisting of the location information and the other of the trip information. Features of the joined dataset inlcude:
- Date/Time
- Location: (latitude, longitude)
- Vehicle starts/stops
- Vehicle idle/stop duration

### Clustering and Visualization

Grouping, clustering and aggregating the data to find the natural downtime. Visualizing the clustered data on a map that shows the most frequented/high duration locations for the vehicle.

![image](https://github.com/user-attachments/assets/6ecf8829-bb80-4a6f-b1f9-c198ec83d071)
![image](https://github.com/user-attachments/assets/c8ffa2ee-e6a5-4b05-bf41-3d763bf79795)

### Recommendation Engine/Model 

Created a recommendation engine that suggest the optimal time and location for booking onsite servicing based on the vehicles natural downtime.

![image](https://github.com/user-attachments/assets/e53a70e3-13d2-409d-a072-709cfa651ca6)

### Predictive LSTM Model

Created an Long Short Term Memory Model in order to predict the location (latitude, longitude) of a vehicle at a given time. Trained on the vehicle data and evaluated on a different vehicle.

Model Results:

- loss: 0.0049
- mae: 0.0500
- Mean Haversine Error: 1907.95 meters
