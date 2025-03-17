import csv
from collections import defaultdict

# Initialize a dictionary to track time spent by each person
time_spent = defaultdict(int)

# Read the CSV and process each row
with open('person_movement.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    # Loop through the rows and count time for each person_id
    for row in reader:
        person_id = row['person_id']
        time_spent[person_id] += 1  # Each entry corresponds to 1 second

# Output the time spent for each person
for person_id, time in time_spent.items():
    print(f"Person {person_id} spent {time} seconds.")
