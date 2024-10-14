# object-prediction-confusion-matrix

## Setup Instructions

1. **Load the model** 

Download the model file (`best.pt`) and rename it as needed. Update the filename in `line 19` of the script.
2. **Load the datasets**

Download the datasets from Roboflow. The dataset should contain the following folders: `train`, `valid`, `test`, and a `data.yaml` file.
You can name the dataset folder anything you want, but make sure to update the folder name in line 8 of the script.
3. **Run the script**

After making the necessary changes, run the script using the following command:
``python confuse.py
``
4. **Output**

The script will generate a CSV file with the format detection_results_<current_date_time>.csv.
5. **Run the generated CSV in Colab**

Using `create_confusion_matrix.ipynb`
Don’t forget to upload the generated .csv file to Colab. Also, update the filename in this line: