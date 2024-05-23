import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Make sure to install Pillow package
import os
from keras.models import load_model
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import keras
from keras.preprocessing import image
import numpy as np
class TomatoClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cucumber Leaf Disease Classification")

        # Title label
        title_label = tk.Label(root, text="Cucumber Leaf Disease Classification", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=10)

        self.image_path = tk.StringVar()

        # Browse Button
        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        # Display selected image
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Submit Button
        self.submit_button = tk.Button(root, text="Classify", command=self.submit_image)
        self.submit_button.pack(pady=10)

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path.set(file_path)
            self.display_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300))  # Resize the image to fit the label
        photo = ImageTk.PhotoImage(image)

        # Update label with the selected image
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def submit_image(self):
        if self.image_path.get():
            # Call your backend function with the image path
            backend_result = classify_tomato(self.image_path.get())
            self.result_label.config(text="Result: " + backend_result)
        else:
            self.result_label.config(text="Please select an image first.")

def classify_tomato(image_path):
    target_size = (224,224)
    model=load_model('confumatrixcucumber.h5')
    print("model loaded")
    dic={0:'Anthracnose',1:'Bacterial Wilt',2:'Downy Mildew',3:'Fresh Leaf',4:'Gummy Stem Blight'
        }
    test_image = load_img(image_path, target_size = (224,224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    result = model.predict(test_image)
    print(np.exp(result))
    print(np.argmax(result))
    result = np.argmax(result)
    detec=dic[result]
    # Placeholder for the backend function
    # You can replace this with your actual classification logic
    return str(detec)

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoClassifierUI(root)
    root.mainloop()
