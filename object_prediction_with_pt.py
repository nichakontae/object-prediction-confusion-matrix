import supervision as sv
from ultralytics import YOLO
import numpy as np
import csv
import datetime

# datasets folder from Roboflow
folder_name = "data"

# Construct the paths with folder_name variable
image_path = f"./{folder_name}/data/images"
label_path = f"./{folder_name}/data/labels"
data_yaml_path = f"./{folder_name}/data.yaml"

# Create the dataset
dataset = sv.DetectionDataset.from_yolo(image_path, label_path, data_yaml_path)

# yolo model
model = YOLO("models/v15m.pt")

index = 0
results = []


def callback(image: np.ndarray) -> sv.Detections:
    # Predict image
    result = model(image)[0]
    global index

    # Initial row element in csv
    row = [index + 1]

    # Initial predict class and actual class
    actual_class = ""
    predict_class = ""
    conf = 0

    # Change image path to label path
    image_path = list(dataset.images.keys())[index]
    index += 1
    label = image_path.replace("images", "labels")
    txt = label.replace(".jpg", ".txt")

    # Open txt file
    try:
        # Open the file in read mode ('r')
        with open(txt, 'r') as file:
            # Read the entire content of the file
            file_content = file.read()

            # Check if the file has content
            if file_content:
                # Split the content by whitespace and get the first element
                first_number_str = file_content.split()[0]

                # Convert 'first_number_str' to an integer
                no = int(first_number_str)

                # Print or use the first number
                print("actual class: ", result.names[no])
                actual_class = result.names[no]
            else:
                # Print a message if the file is empty
                print("actual class: background")
                actual_class = "Background"
    except FileNotFoundError:
        print(f"File '{txt}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    row.append(actual_class)
    # Get information from prediction
    detections = sv.Detections.from_ultralytics(result)
    if detections:
        no = detections.class_id[0]
        raw_conf = detections.confidence[0]
        print("predict class: ", result.names[no])
        print("conf: ", round(raw_conf, 2))
        conf = round(raw_conf,2)
        predict_class = result.names[no]
    else:
        print("predict class: background")
        predict_class = "Background"

    # append predict class, conf
    row.append(predict_class)
    row.append(conf)

    # append row data in to all result
    results.append(row)
    print("results: ",results)
    return sv.Detections.from_ultralytics(result)


confusion_matrix = sv.ConfusionMatrix.benchmark(
    dataset=dataset,
    callback=callback
)

confusion_matrix.plot()

# Create a CSV file for writing the results

# Base file name
base_filename = "detection_results"

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format the current date and time as a string (e.g., "YYYY-MM-DD_HH-MM-SS")
timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Combine the base filename and timestamp to create a unique filename
csv_filename = f"{base_filename}_{timestamp_str}.csv"



with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header row
    header = ["No.", "Actual_Class", "Predicted_Class", "Confidence"]
    csv_writer.writerow(header)

    for row in results:
        # Write the result to the CSV file
        csv_writer.writerow(row)

print(f"Results written to {csv_filename}")
