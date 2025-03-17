#track.py

#track.py
class TrackState:
    """
    Enumeration type for the single target track state.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated velocities.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, class_name=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
            print(f"Initialized track_id {self.track_id} with feature. Total features: {len(self.features)}")

        self._n_init = n_init
        self._max_age = max_age
        self.class_name = class_name

    def to_tlwh(self):
        """Convert bounding box to `(top left x, top left y, width, height)` format."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Convert bounding box to `(min x, min y, max x, max y)` format."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_class(self):
        """Return the class name of the object."""
        return self.class_name

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a Kalman filter."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, global_database):
        """Perform Kalman filter measurement update step and update the feature cache."""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        
        # Append the detection's feature to the features list
        if detection.feature is not None:
            self.features.append(detection.feature)  # Keep only the latest feature
            print(f"Adding feature to track_id {self.track_id}. Total features: {len(self.features)}")
            
            # Update the global database only if the track is Confirmed
            if self.state == TrackState.Confirmed:
                if self.track_id in global_database:
                    global_database[self.track_id]["features"].append(detection.feature)
                else:
                    global_database[self.track_id] = {"features": [detection.feature], "class_name": self.class_name}
                print(f"Updated global database for track_id {self.track_id} with {len(global_database[self.track_id]['features'])} features.")
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        print(f"Checking track_id {self.track_id} for deletion. State: {self.state}, time_since_update: {self.time_since_update}, max_age: {self._max_age}")
        
        # Check if the track should be deleted
        if self.state == TrackState.Tentative:
            # Tentative tracks are deleted without being added to the global database
            self.state = TrackState.Deleted
            print(f"Track_id {self.track_id} marked as Deleted (Tentative).")
        elif self.time_since_update >= self._max_age and self.state == TrackState.Confirmed:
            # Only Confirmed tracks are added to the global database before deletion
            print(f"Track_id {self.track_id} has {len(self.features)} features to add to the global database.")
            self.state = TrackState.Deleted
            print(f"Track_id {self.track_id} marked as Deleted (max_age exceeded).")
        else:
            print(f"Track_id {self.track_id} not deleted. State: {self.state}, time_since_update: {self.time_since_update}, max_age: {self._max_age}")

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

