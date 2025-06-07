import os

def read_labels_from_file(filepath):
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:
                labels.append(tuple(parts))
    return labels

def is_side_clear(labels, side, threshold=0.05):
    total_area = 0
    for _, x_center, _, width, height in labels:
        if side == 'left' and x_center < 0.4: 
            total_area += width * height
        elif side == 'right' and x_center > 0.6:
            total_area += width * height
    return total_area < threshold

#I supposed center is between x=0.4 and x=0.6
def decide_direction(labels):
    left = right = middle = 0
    for label in labels:
        _, x_center, _, _, _ = label
        if 0.4 <= x_center <= 0.6:
            middle += 1
        elif x_center < 0.4:
            left += 1
        else:
            right += 1

    if middle > 0:
        left_clear = is_side_clear(labels, 'left')
        right_clear = is_side_clear(labels, 'right')
        
        if left < right and left_clear:
            return "left"
        elif right_clear:
            return "right"
        elif left_clear:
            return "left"
        else:
            return "stop"
    else:
        return "straight"

# Run on all label files
label_dir = 'labels/'  # path to your label directory
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        labels = read_labels_from_file(filepath)
        decision = decide_direction(labels)
        print(f"{filename}: {decision}")
