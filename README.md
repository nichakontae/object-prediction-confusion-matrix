# Object Prediction Confusion Matrix

If you train the model on Google Colab, you will get a model with a `.pt` file, so you need to use `object_prediction_with_pt.py`. 

However, if you train the model on Roboflow, please use `object_prediction_with_roboflow.py`.
## Install Required Packages

Run the following command to install all the necessary packages:

```bash
pip install -r requirements.txt
```

## Setup Instructions with `object_prediction_with_pt.py`

1. **Load the Model**  
   Download the model file (`best.pt`) and rename it as needed. Update the filename in **line 19** of the script. The model `.pt` file will be in `runs/detect/train/weights/best.py`

2. **Load the Datasets**  
   Download the datasets from Roboflow. The dataset should contain the following folders: `train`, `valid`, `test`, and a `data.yaml` file.  
   You can name the dataset folder anything you want, but make sure to update the folder name in **line 8** of the script.

3. **Run the Script**  
   After making the necessary changes, run the script using the following command:

   ```bash
   python confuse.py
   ```

4. **Output**  
   The script will generate a CSV file in the format: `detection_results_<current_date_time>.csv`.

5. **Run the Generated CSV in Colab**  
   Use the provided `create_confusion_matrix.ipynb` notebook in Colab to analyze the CSV file.  
   Don’t forget to upload the generated `.csv` file to Colab, and make sure to update the filename at the following line:

   ```python
   model_pred = pd.read_csv('detection_results_<your_csv_file>.csv')
   ```

## Setup Instructions with `object_prediction_with_roboflow.py`

1. **Load the Datasets**  
   Download the datasets from Roboflow. The dataset should contain the following folders: `train`, `valid`, `test`, and a `data.yaml` file.  
   You can name the dataset folder anything you want, but make sure to update the folder name in **line 8** of the script.

2. **Copy Private API Key**  
   Copy the Private API Key for your project from the Roboflow dashboard.
   <img width="1013" alt="Screenshot 2567-10-14 at 23 24 21" src="https://github.com/user-attachments/assets/0b5ececf-ba03-45c3-8f6b-6bdd89a34c1a">


4. **Update the Project Name**  
   Ensure that you use the correct project name in the script, and confirm that the API key corresponds to the workspace of your current project.

5. **Select the Model Version**  
   Choose the appropriate model version number for your project. For example, in this case, use version number `36`.
   <img width="281" alt="Screenshot 2567-10-14 at 23 30 59" src="https://github.com/user-attachments/assets/917c74bb-6ebb-41a4-b853-819277c6bf59">


7. **Run the Script**  
   Feel free to select your preferred confidence score.
   After making the necessary changes, run the script using the following command:

   ```bash
   python confuse.py
   ```

8. **Output**  
   The script will generate a CSV file in the format: `detection_results_<current_date_time>.csv`.

9. **Run the Generated CSV in Colab**  
   Use the provided `create_confusion_matrix.ipynb` notebook in Colab to analyze the CSV file.  
   Don’t forget to upload the generated `.csv` file to Colab, and make sure to update the filename at the following line:

   ```python
   model_pred = pd.read_csv('detection_results_<your_csv_file>.csv')
   
