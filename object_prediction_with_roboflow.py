import supervision as sv
from roboflow import Roboflow
import numpy as np
import csv
import datetime
import yaml

# Initialize Roboflow model
rf = Roboflow(api_key="PRIVATE_API_KEY")
project = rf.workspace().project("PROJECT_ID")
roboflow_model = project.version(version_number).model

# Dataset paths from Roboflow
folder_name = "data"
image_path = f"./{folder_name}/test/images"
label_path = f"./{folder_name}/test/labels"
data_yaml_path = f"./{folder_name}/data.yaml"

print("Image Path:", image_path)
print("Label Path:", label_path)
print("Data YAML Path:", data_yaml_path)

# Load the data.yaml to get class names
with open(data_yaml_path, 'r') as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)
class_names = data_yaml['names']

# Create the dataset (images and labels)
dataset = sv.DetectionDataset.from_yolo(image_path, label_path, data_yaml_path)

results = []

def callback(image: np.ndarray, image_file: str) -> sv.Detections:
    row = [len(results) + 1]  # Use length of results for the current index
    print(f"{row[0]}: {image_file}")
    actual_class = ""
    predict_class = ""
    conf = 0

    # Perform inference with Roboflow model
    predictions = roboflow_model.predict(image_file, confidence=40, overlap=30).json()

    # Process the prediction results
    if predictions["predictions"]:
        predicted_class = predictions["predictions"][0]["class"]
        conf = predictions["predictions"][0]["confidence"]
        conf = round(conf, 2)
        print("Predict class: ", predicted_class)
        print("Confidence: ", round(conf, 2))
        predict_class = predicted_class
    else:
        print("Predict class: background")
        predict_class = "Background"

    # Derive the label file path from the image file path
    label = image_file.replace("images", "labels")
    txt = label.replace(".jpg", ".txt")

    # Open the corresponding label file
    try:
        with open(txt, 'r') as file:
            file_content = file.read()
            if file_content:
                actual_class_id = int(file_content.split()[0])
                actual_class = class_names[actual_class_id]
                print("Actual class: ", actual_class)
            else:
                actual_class = "Background"
                print("Actual class: background")
    except FileNotFoundError:
        print(f"File '{txt}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Add results to row (actual and predicted)
    row.append(actual_class)
    row.append(predict_class)
    row.append(conf)

    # Add the row to the results
    results.append(row)

    # Return empty detections, since Supervision expects a return
    return sv.Detections.empty()

# Iterate through the dataset without accessing .images directly
for image_file, image, annotation in dataset:
    print(f"Processing: {image_file}")  # Print the image file being processed
    callback(image, image_file)

# Generate confusion matrix
def benchmark_callback(image: np.ndarray) -> sv.Detections:
    # Get the current index based on the results length
    current_index = len(results)

    if current_index < len(dataset):  # Check if the index is within the dataset range
        image_file = dataset[current_index][0]  # Access the image file based on the current index
        return callback(image, image_file)
    else:
        return sv.Detections.empty()  # Return empty detections if index is out of range


confusion_matrix = sv.ConfusionMatrix.benchmark(
    dataset=dataset,
    callback=benchmark_callback  # Use the defined benchmark callback function
)

confusion_matrix.plot()

# Save results to CSV
base_filename = "detection_results"
current_datetime = datetime.datetime.now()
timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"{base_filename}_{timestamp_str}.csv"

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ["No.", "Actual_Class", "Predicted_Class", "Confidence"]
    csv_writer.writerow(header)

    for row in results:
        csv_writer.writerow(row)

print(f"Results written to {csv_filename}")
