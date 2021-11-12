# Importing GUI modules
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# importing modules for predictions
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# using the frontal face harr cascade file from OpenCV (opencv/opencv, 2013).
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# loading the trained model
model = load_model('trained_model.h5')

# Listing out the category classes for classification
classes = ['Angry', 'Disgust','Fear','Happy','Neutral','Sad','Surprise']

# The following function does the classification of the imahe 
def classifyImages(file_path):
  image = cv2.imread(file_path)
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = classifier.detectMultiScale(grayImage, 1.2, 4)
  # after detecting all the faces in the image  
  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (231, 76, 60), 2)
    gray = grayImage[y:y+h, x:x+w]
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    finalImage = gray.astype('float')/255.0
    finalImage = img_to_array(finalImage)
    finalImage = np.expand_dims(finalImage, axis=0)

    preds = model.predict(finalImage)[0]
    label = classes[preds.argmax()]
    label_position = (x, y)
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (52, 152, 219), 2)
  
  # Converting image array to image  
  im = Image.fromarray(image)
  imgtk = ImageTk.PhotoImage(image=im) 
  
  # Displayng the labeled image to the user
  displayLabel.config(image=imgtk)
  displayLabel.image = imgtk

# uploadImage function works after clicking on the uploadButton and displays the image 
def uploadImage():
  fileName = filedialog.askopenfilename()
  uploadedImage = Image.open(fileName)
  uploadedImage.thumbnail((380, 500))
  imageFile = ImageTk.PhotoImage(uploadedImage)
  displayLabel.config(image=imageFile)
  displayLabel.image = imageFile
  # Using the image classification function
  classifyImages(fileName)

# Exitind the app after cliking the closeApp button
def closeApp():
  exit()

# Creating the GUI in Tkinter
root = tk.Tk()

appFrame = tk.Frame(root)
appFrame.pack(side=tk.BOTTOM, padx=15, pady=15)

# Label to display the image
displayLabel = tk.Label(root)
displayLabel.pack(padx=15, pady=15)

uploadButton = tk.Button(appFrame, text="Upload an Image", command=uploadImage)
uploadButton.pack(side=tk.LEFT)

closeButton = tk.Button(appFrame, text="Close App", command=closeApp)
closeButton.pack(side=tk.LEFT, padx=10)

root.title("Facial Expression Image Classification")
root.geometry("400x580")
root.mainloop()
