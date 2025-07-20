# Facial_Emotion_recognition 😊😢

=> This project involves image loading and preprocessing, face detection using Haar Cascades, applying transfer learning with a MobileNetV2 model for classification. 

Focuses on training a Convolutional Neural Network (CNN) to classify images into 7 categories using augmented image data and saving the trained model, and making predictions on new images.

After that using pre-trained deep learning model and OpenCV this project performs real-time face emotion recognition.

Here's a breakdown of the key steps:

   Data Loading and Preprocessing: The code sets up ImageDataGenerator for training and validation datasets. It specifies the directories containing the images, targets image size and color mode (grayscale), and includes data augmentation techniques for the training set.
   
   Model Architecture: A Sequential CNN model is defined with multiple convolutional, pooling, batch normalization, and dropout layers. The model ends with a dense layer with 7 units and a softmax activation, indicating it's designed to classify images into 7 categories.
   
   Model Compilation: The model is compiled using the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.
   
   Model Training: The model is trained using the fit method, utilizing the image data generators to feed data in batches. The training is run for 20 epochs.
   
   Model Saving: The trained model is saved to an HDF5 file.

   
   Image Loading and Preprocessing: Loads images, resizes them, and prepares them for a deep learning model.
    
   Face Detection: Uses a Haar Cascade classifier to detect faces within images.
    
   Transfer Learning: Utilizes a pre-trained MobileNetV2 model and adds custom layers for a specific   task (likely image classification based on the output layer with 7 classes).
    
   Prediction: Makes predictions on a new image using the trained model.

