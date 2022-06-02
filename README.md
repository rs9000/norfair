![Norfair by Tryolabs logo](docs/logo.png)

Fork of https://github.com/tryolabs/norfair <br>
Customized for Maritime Vessel Tracking.

## Features

- Detection class support **MMSI** and **message_type** (ex. AIS)
- The distance function is matrix, for fast computation.

## Installation

```bash
git clone https://github.com/rs9000/norfair
```

## How it works

Define a matricial distance metrics

```python
def matricial_euclidean(threshold, detections, objects):
    detections_arr = np.array([x.points for x in detections])
    objects_arr = np.array([x.estimate[0] for x in objects])
    same_mmsi = np.array([x.mmsi == y.mmsi for x in detections for y in objects]).reshape(len(detections), len(objects))

    distance_matrix = scipy.spatial.distance_matrix(detections_arr, objects_arr)
    distance_matrix[distance_matrix > threshold] = threshold + 1
    distance_matrix[same_mmsi] = 0

    return distance_matrix
```

## Tracking Example

```python
class Message():
    def __init__(self, timestamp, position, stn, mmsi, type):
        self.timestamp = timestamp
        self.position = position
        self.mmsi = mmsi
        self.type = type


tracker = Tracker(distance_function=matricial_euclidean, distance_threshold=0.5,
                  initialization_delay=3)

messages_t0 = [Message("987298479", np.array([44, 22]), "873624", "AIS"),
            Message("987298479", np.array([44, 22]), "873624", "AIS")]

messages_t1 = [Message("987298479", np.array([44, 22]), "873624", "AIS"),
            Message("987298479", np.array([44, 22]), "873624", "AIS")]

messages_t2 = [Message("987298479", np.array([44, 22]), "873624", "AIS"),
            Message("987298479", np.array([44, 22]), "873624", "AIS")]


messages = [messages_t0, messages_t1, messages_t2]

for t in messages:
    detections = [Detection(points=np.array([x.position for x in t]),
                            mmsi=[x.mmsi for x in t],
                            message_type=[x.message_type for x in t])]

    tracked_objects = tracker.update(detections=detections)
```

## Citing Norfair

For citations in academic publications, please export your desired citation format (BibTeX or other)
from [Zenodo](https://doi.org/10.5281/zenodo.5146253).

## License

Copyright © 2022, [Tryolabs](https://tryolabs.com). Released under the [BSD 3-Clause](LICENSE).
