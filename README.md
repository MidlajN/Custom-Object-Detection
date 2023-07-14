# Custom-Object-Detection using YOLOv5

This repository provides Python programs and instructions to create a custom object detection model using YOLOv5,
a popular deep learning framework for real-time object detection. The model can be trained on your own dataset and
used to detect and localize objects of interest in images or videos. The programs included here will help you
in the following tasks:
<br><br>
<b> 1. Dataset Creation: </b> Use the `dataset_creation.py` program to create a custom dataset for training your object detection
model. The program allows you to capture images from a camera, annotate the objects of interest, and save the 
dataset in the required format for YOLOv5.<br><br>
<b> 1. Model Training: </b> Train the YOLOv5 model on the custom dataset using the `training_dataset.py` program. This program
will take care of configuring the model architecture, loading the dataset, setting hyperparameters, and saving the
trained model weights.<br><br>
<b> 3. Object Detection: </b> Utilize the `detection.py` program to perform object detection on images or videos
using the trained model. The program takes an input image or video file, applies the object detection algorithm, and
displays the results with bounding boxes and labels around the detected objects.<br><br>

## Prerequisites
Before Using These Programs, Ensure That You Have The Following Prerequisites Installed :
<ul>
  <li>Python 3.6 or later </li>
  <li>PyTorch 1.7 or later</li>
  <li>OpenCV</li>
</ul>

## Getting Started
1. Clone this repository to your local machine.
2. Install the required dependencied by running `pip install -r requirements.txt` in your terminal or command prompt
3. Run the `dataset_creation.py` program and select the Region Of Interest where the object is.
4. Start the training process by running the `dataset_training.py` program. Monitor the training progress and save the trained weights oncce the desired performance is achieved.
5. To perform object detection on new images or videos, use the `detection.py` program. Provide path to the weight made by the `dataset_training.py`

Feel Free to modify and adapt these programs according to your specific requirements

## Contributions
Contributions to this repository are welcome! If you encounter any issues or have suggestions for improvements,
please submit an issue or create a pull request. Your feedback and contributions will help make this project
better for everyone.

## Acknowledgements
I would like to express our gratitude to the creators and contributors of the YOLOv5 project for their
remarkable work, which served as the foundation for this custom object detection solution.

## References
<ul>
  <li>YOLOv5 GitHub repository: https://github.com/ultralytics/yolov5</li>
  <li>YOLOv5 Documentation: https://ultralytics.com/yolov5</li>
</ul><br><br> 
 
> *If there any known bugs found or run into any issues please let me know. Please enjoy and feel free to share
  your opinion, constructive criticism, or comments about my work. Email:  midlaj4n@gmail.com  Thank you!*

