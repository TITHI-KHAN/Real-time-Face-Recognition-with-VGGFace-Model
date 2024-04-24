# Real-time-Face-Recognition-with-VGGFace-Model

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### Thank you.

<br>

## Code Explanation:

1. **Importing Libraries**: The code imports necessary libraries including `cv2` for OpenCV operations, `os` for file operations, `numpy` for numerical computations, and specific modules from `keras_vggface` and `keras.preprocessing` for working with VGGFace model and preprocessing images.

2. **Loading VGGFace Model**: It loads the VGGFace model with ResNet50 architecture (`model='resnet50'`) pre-trained on the 'vggface2' dataset. This model is used for feature extraction.

3. **Loading Haar Cascade Classifier**: It loads the Haar cascade classifier for face detection from OpenCV's pre-trained models.

4. **Feature Extraction**: The function `extract_features()` preprocesses and extracts features from an input image using the loaded VGGFace model.

5. **Loading Known Faces**: It iterates through the directories in the 'training_data' directory, extracts features from the first image of each person (assuming one image per person), and stores the features along with the person's name in the `known_faces` dictionary.

6. **Capturing Video**: It initializes a video capture object using `cv2.VideoCapture(0)` to capture frames from the default camera (index 0).

7. **Real-time Face Recognition Loop**: Inside the main loop, it continuously captures frames from the video feed, converts each frame to grayscale, and detects faces using the Haar cascade classifier.

8. **Recognizing Faces**: For each detected face, it extracts features, calculates the cosine similarity between the captured features and the features of known faces, and identifies the person with the minimum cosine distance. It draws rectangles around the detected faces and displays the predicted identity along with the cosine distance.

9. **Exiting**: The program exits the loop if the 'q' key is pressed, releasing the video capture and closing all OpenCV windows.


## Key Points:
- Utilizes VGGFace model pre-trained on ResNet50 architecture for feature extraction.
- Loads Haar cascade classifier for face detection.
- Extracts features from known faces and stores them for recognition.
- Performs real-time face recognition using cosine similarity.
- Displays recognized faces with bounding boxes and predicted identities along with cosine distances.
- Allows quitting the application by pressing the 'q' key.
