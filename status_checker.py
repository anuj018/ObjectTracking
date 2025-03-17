import math
import logging

class UnionFind:
    """Union-Find data structure for clustering people based on proximity."""
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

# Set up logging
log_filename = "status_checker_logs.txt"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def improved_human_status(
    track, current_time, recent_entries, previous_positions, entry_line_y,
    sliding_window=6.0, distance_threshold=100.0, min_frames_for_entry=20
):
    """
    Determines human status using entry line validation and sliding window finalization.
    Returns tuple: (classification, is_final)
    """
    track_id = track.track_id
    x1, y1, x2, y2 = track.to_tlbr()

    # Check if already classified
    if track_id in recent_entries.get("classified", {}):
        status = recent_entries["classified"][track_id]
        return (status, True)  # Return as already finalized

    # Initialize data structures
    recent_entries.setdefault("entries", [])
    recent_entries.setdefault("classified", {})
    recent_entries.setdefault("cluster_ids", {})  # maps cluster_key to a unique number
    recent_entries.setdefault("cluster_counter", 0)


    # Entry validation: y1 must be below entry line
    valid_entry = y1 < entry_line_y
    entry = next((e for e in recent_entries["entries"] if e["track_id"] == track_id), None)

    # New entry creation
    if not entry:
        valid_entry = y1 < entry_line_y
        if valid_entry:
            print(f"valid entry for {track_id}")
            entry = {
                "track_id": track_id,
                "start_time": current_time,
                "last_time": current_time,
                "positions": [(x1, y1, x2, y2)],
                "max_cluster_size": 1,
                "frame_count": 1,
                "cluster_key": None
            }
            recent_entries["entries"].append(entry)
            logging.info(f"New entry created for Track ID {track_id}")
        else:
            # Mark as invalid and return immediately.
            recent_entries["classified"][track_id] = "invalid"
            logging.info(f"Track ID {track_id} rejected due to invalid entry (y1: {y1} is not less than {entry_line_y})")
            return (None, False)

    # Update tracking information
    previous_positions[track_id] = y1
    if entry:
        entry["last_time"] = current_time
        entry["positions"].append((x1, y1, x2, y2))
        entry["frame_count"] += 1

    # Finalize classification when sliding window expires
    if entry and (current_time - entry["start_time"]) >= sliding_window:
        classification = "alone" if entry["max_cluster_size"] == 1 else \
                        "couple" if entry["max_cluster_size"] == 2 else "group"
        if classification in ["couple", "group"]:
            cluster_key = entry.get("cluster_key")
            if cluster_key:
                if cluster_key not in recent_entries["cluster_ids"]:
                    recent_entries["cluster_counter"] += 1
                    recent_entries["cluster_ids"][cluster_key] = recent_entries["cluster_counter"]
                cluster_id = recent_entries["cluster_ids"][cluster_key]
                classification = f"{classification}_{cluster_id}"
        recent_entries["classified"][track_id] = classification
        recent_entries["entries"].remove(entry)
        print(f"finalizing status of {track_id} to {classification}")
        return (classification, True)

    # Prune old entries
    recent_entries["entries"] = [e for e in recent_entries["entries"]
                               if (current_time - e["last_time"]) <= sliding_window]

    # Cluster analysis for provisional classification
    if entry and entry["frame_count"] >= min_frames_for_entry:
        print(f"Person {track_id} visible for sufficient number of frames")
        n = len(recent_entries["entries"])
        uf = UnionFind(n)
        
        # Create proximity clusters
        for i in range(n):
            for j in range(i+1, n):
                x_i, y_i = recent_entries["entries"][i]["positions"][-1][:2]
                x_j, y_j = recent_entries["entries"][j]["positions"][-1][:2]
                if math.hypot(x_j-x_i, y_j-y_i) <= distance_threshold:
                    uf.union(i, j)

        # Update cluster sizes
        clusters = {}
        for idx in range(n):
            root = uf.find(idx)
            clusters.setdefault(root, []).append(recent_entries["entries"][idx]["track_id"])
        
        for idx in range(n):
            entry_i = recent_entries["entries"][idx]
            for root, track_ids in clusters.items():
                if entry_i["track_id"] in track_ids:
                    if len(track_ids) > 1:
                        cluster_key = "_".join(map(str, sorted(track_ids)))
                        entry_i["cluster_key"] = cluster_key
                    else:
                        entry_i["cluster_key"] = None
                    break

        # Find current cluster and update sizes
        current_cluster = next((c for c in clusters.values() if track_id in c), None)
        if current_cluster:
            cluster_size = len(current_cluster)
            for member_id in current_cluster:
                member_entry = next(e for e in recent_entries["entries"] if e["track_id"] == member_id)
                member_entry["max_cluster_size"] = max(member_entry["max_cluster_size"], cluster_size)

        # Return provisional classification
        current_size = entry["max_cluster_size"]
        return (
            "alone" if current_size == 1 else
            "couple" if current_size == 2 else
            "group", False
        )
    
    print(f'current frame count for {track_id} is {entry["frame_count"]}')

    return (None, False)