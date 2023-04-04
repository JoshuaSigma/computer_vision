import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import helper as h

st.title("Computer Vision")
st.text("Josh Patterson  |  March 23, 2023")

st.image("https://uploads-ssl.webflow.com/614c82ed388d53640613982e/635bcc320bd72c8dff981f27_634fd7a0d519102536f56469_6320785ed40274f7631b761c_what-is-computer-vision.png")

with st.container():
    st.write("In today's world, computer vision technology has become an essential tool for a variety of applications, ranging from autonomous vehicles to surveillance systems. Two important areas of computer vision are face detection and object detection with image segmentation. In this article, we will explore two popular frameworks in computer vision, OpenCV and TensorFlow, and how they can be used to perform these tasks. Specifically, we will showcase the power of OpenCV for face detection and TensorFlow for object detection and image segmentation. With the help of these frameworks, developers and researchers can build more robust and accurate computer vision systems that can be used in a variety of applications, including security, healthcare, and retail. So, let's dive into the world of computer vision and explore these powerful tools in action.")

color = [{"name":"Red","hex":"ff0000","rgb":[255,0,0],"cmyk":[0,100,100,0],"hsb":[0,100,100],"hsl":[0,100,50],"lab":[53,80,67]},{"name":"Green","hex":"00ff00","rgb":[0,255,0],"cmyk":[100,0,100,0],"hsb":[120,100,100],"hsl":[120,100,50],"lab":[88,-86,83]},{"name":"Fuchsia","hex":"ff00ff","rgb":[255,0,255],"cmyk":[0,100,0,0],"hsb":[300,100,100],"hsl":[300,100,50],"lab":[60,98,-61]},{"name":"Tropical indigo","hex":"8080ff","rgb":[128,128,255],"cmyk":[50,50,0,0],"hsb":[240,50,100],"hsl":[240,100,75],"lab":[59,33,-63]},{"name":"Cyan (RGB)","hex":"00ffff","rgb":[0,255,255],"cmyk":[100,0,0,0],"hsb":[180,100,100],"hsl":[180,100,50],"lab":[91,-48,-14]},{"name":"Blue","hex":"0000ff","rgb":[0,0,255],"cmyk":[100,100,0,0],"hsb":[240,100,100],"hsl":[240,100,50],"lab":[32,79,-108]},{"name":"Navy blue","hex":"00007e","rgb":[0,0,126],"cmyk":[100,100,0,51],"hsb":[240,100,49],"hsl":[240,100,25],"lab":[13,47,-64]},{"name":"Office green","hex":"007d00","rgb":[0,125,0],"cmyk":[100,0,100,51],"hsb":[120,100,49],"hsl":[120,100,25],"lab":[45,-51,49]},{"name":"Dark red","hex":"860000","rgb":[134,0,0],"cmyk":[0,100,100,47],"hsb":[0,100,53],"hsl":[0,100,26],"lab":[27,50,40]},{"name":"Gunmetal","hex":"1b3335","rgb":[27,51,53],"cmyk":[49,4,0,79],"hsb":[185,49,21],"hsl":[185,33,16],"lab":[19,-9,-4]}]

# External tools
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

default_img_path = "https://i.ibb.co/wczXsrn/IMG-20230319-052913.jpg"
default_img = h.load_img(default_img_path)


# OpenCV Text
st.subheader("OpenCV")
st.write("OpenCV's Cascade Classifier is a powerful tool for detecting objects in images and videos. It is particularly useful for face detection, which involves locating and identifying human faces in images and videos. The cascade classifier is a machine learning algorithm that is trained to identify certain features of an object, such as the edges of a face, and use these features to identify the object.")
st.write("One of the advantages of the cascade classifier and detectMultiScale functions is that they are fast and efficient, allowing for real-time face detection in video streams. This makes them ideal for applications such as security systems and video conferencing, where real-time detection is essential.")
st.write("The detectMultiScale function in OpenCV is used to apply the cascade classifier to an image or video. It works by scanning the image or video at different scales, looking for areas that match the features identified by the classifier. When a match is found, the function returns a set of coordinates that define the location of the object in the image or video.")

st.subheader("CODE")
st.write("After we import the necessary packages let's setup a function to obtain the data from an image path on the web. If you are uploading an image, you can skip this.")
st.code('''import numpy as np
import cv2
from PIL import Image

# Function to get image from web path
def load_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
''')
st.write("Here we define the location of our classifier, 'face_cascade,' and pass our path to path to the function. Once we have the data, we convert it to a numpy array. And convert to grayscale to deliver it to cv2 (OpenCV).")
st.code('''# Lets define our Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set image path and call Function to get image
default_img_path = "https://i.ibb.co/wczXsrn/IMG-20230319-052913.jpg"
default_img = load_img(default_img_path)

# Convert image to numpy
numpy_image = np.array(default_image)

# Grayscale is required for this classifier, let's convert
gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
''')
st.write("Now, for the magic. We pass our grayscale numpy array and some variables to detectMultiScale. Scale factor, minimum neighbors, min and max size can be adjusted. The distance from the focus subjects to the camera (Zoom) will affect the process. See demo below.")
st.code('''# Detect faces
faces = face_cascade.detectMultiScale(gray, a, b, c)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(numpy_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
numpy_image

# Back to image to draw boxes on it
img = Image.fromarray(numpy_image)

# Crop Faces out
if faces is not None:
    for (x, y, w, h) in faces:
        
        # Crop out images and display
        img.crop((x, y, (x+w), (y+h)))
''')
st.write("We can take the output we get for faces and draw boxes or anything else you need, cropping them out, etc.")

# Image upload
st.subheader("Upload your image to use here for OpenCV only")
image_file = st.file_uploader("You can replace the image with one of your images here.", type=['jpg', 'png', 'jpeg'])
if image_file is None:
    original_image = default_img
    numpy_image = np.array(original_image)
else:
    original_image = Image.open(image_file)
    numpy_image = np.array(original_image)

# Grayscale for comp vision
gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

st.write("Try changing the args for detectMultiScale below and see how it alters the output.")

# Columns set locations
col0, col1, col2 = st.columns(3)

# Assign cols
a = col0.slider("Scale Factor", min_value=1.02, max_value=1.26, value=1.07)
b = col1.slider("Minimum Neighbors", min_value=1, max_value=12, value=4)
c = col2.slider("Minimum Size", min_value=5, max_value=50, value=30)

# Detect faces
faces = face_cascade.detectMultiScale(gray, a, b, c)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(numpy_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
st.image(numpy_image)

# Back to image to draw boxes on it
img = Image.fromarray(numpy_image)

# Crop Faces out
if faces is not None:
    for (x, y, w, h) in faces:
        
        st.image(img.crop((x, y, (x+w), (y+h))))

# Tensflow Object Detection ------------------------------------------------------------------------------------------

st.subheader("Object Detection with Tensorflow")
st.write("TensorFlow is a popular open-source framework for building and training machine learning models, including those used in object detection. TensorFlow's Object Detection API is a collection of pre-trained models that can be fine-tuned for specific applications, as well as tools for training new models from scratch.")
st.write("The pre-trained model we will look at, available in the Object Detection API, is the CenterNet Hourglass 512x512 model. This model is based on the CenterNet architecture, which is a state-of-the-art object detection framework that uses keypoint estimation to locate objects in an image.")
st.write("The Hourglass 512x512 model is trained on a large dataset of images and can detect a wide range of objects, including people, animals, and vehicles. It uses a deep neural network to analyze an image and identify the location and size of each object in the image. This information is then used to draw bounding boxes around the objects, which can be used for further analysis or classification.")
st.write("Using the CenterNet Hourglass 512x512 model is easy, thanks to its availability on TensorFlow Hub. With just a few lines of code, developers and researchers can use the model to detect objects in images or videos. Additionally, the model can be fine-tuned for specific applications by training it on a new dataset of images.")
st.subheader("CODE")
st.write("We need to import tensorflow, tensorflow_hub, numpy and pillow. Always make sure you have the packages installed in your environment. Define a function to convert an image to a tensor.")
st.code('''import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Function to convert image into tensor
def tf_preproc(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[:, :, :, :3]
    return input_tensor
''')
st.write("Next, let's load the model. We need to preprocess our image into a tensor. Then assign our output variables. We are given coded detection classes")
st.code('''# Load the model and give the preprocessed image to the model, get outputs
detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
detector_output = detector(tf_preproc(img))
class_ids = tf.squeeze(detector_output["detection_classes"])
scores = tf.squeeze(detector_output["detection_scores"])
boxes = tf.squeeze(detector_output["detection_boxes"])
''')
st.write("Tf_classes.txt contains the mappings for the class names. Lets use it to decode the class ids.")
st.code('''# Empty list to hold class names
class_names = []

# Use detection classes to map to tf_classes, we are mapping the return codes to the correct class.
# Example class 1 is person. So we map all 1's to the word person, 2 is bicycle, etc. 
with open('tf_classes.txt', 'r') as f:
    input_lines = f.readlines()
    for line in input_lines:
        line = line.strip()
        if line.startswith("display_name:"):
            class_names.append(line.split()[1].replace('"', ''))
''')
st.write("Let's set max_objects to 5. This way we can see the top 5 detection scores and relative boxes. Also lets make a copy of the original image to draw boxes on, And make it a numpy array.")
st.code('''# Define the number of objects to display info on in the image
# Remember you get all objects returned, depending on the application this will vary
max_objects = 5

# Convert to numpy arrays
boxes = boxes.numpy()
tf_image = np.array(original_image)
''')
st.write("Finally, we draw our objects' boxes and labels, overlay the text onto the image and display it.")
st.code('''# Pick colors for boxes
color = [{"name":"Red","hex":"ff0000","rgb":[255,0,0],"cmyk":[0,100,100,0],"hsb":[0,100,100],"hsl":[0,100,50],"lab":[53,80,67]},{"name":"Green","hex":"00ff00","rgb":[0,255,0],"cmyk":[100,0,100,0],"hsb":[120,100,100],"hsl":[120,100,50],"lab":[88,-86,83]},{"name":"Fuchsia","hex":"ff00ff","rgb":[255,0,255],"cmyk":[0,100,0,0],"hsb":[300,100,100],"hsl":[300,100,50],"lab":[60,98,-61]},{"name":"Tropical indigo","hex":"8080ff","rgb":[128,128,255],"cmyk":[50,50,0,0],"hsb":[240,50,100],"hsl":[240,100,75],"lab":[59,33,-63]},{"name":"Cyan (RGB)","hex":"00ffff","rgb":[0,255,255],"cmyk":[100,0,0,0],"hsb":[180,100,100],"hsl":[180,100,50],"lab":[91,-48,-14]},{"name":"Blue","hex":"0000ff","rgb":[0,0,255],"cmyk":[100,100,0,0],"hsb":[240,100,100],"hsl":[240,100,50],"lab":[32,79,-108]},{"name":"Navy blue","hex":"00007e","rgb":[0,0,126],"cmyk":[100,100,0,51],"hsb":[240,100,49],"hsl":[240,100,25],"lab":[13,47,-64]},{"name":"Office green","hex":"007d00","rgb":[0,125,0],"cmyk":[100,0,100,51],"hsb":[120,100,49],"hsl":[120,100,25],"lab":[45,-51,49]},{"name":"Dark red","hex":"860000","rgb":[134,0,0],"cmyk":[0,100,100,47],"hsb":[0,100,53],"hsl":[0,100,26],"lab":[27,50,40]},{"name":"Gunmetal","hex":"1b3335","rgb":[27,51,53],"cmyk":[49,4,0,79],"hsb":[185,49,21],"hsl":[185,33,16],"lab":[19,-9,-4]}]
colors_picked = []

# Draw boxes (Expanded for brevity)
# Loop through the height
for i in range(max_objects):
    # Loop through the width
    for j in range(len(boxes[i])):
        # If its 0 then its the height
        if j == 0:
            boxes[i][j] = round(boxes[i][j] * original_image.height)
        # Else its the width
        else:
            boxes[i][j] = round(boxes[i][j] * original_image.width)
    # Plug our numbers into correct spots and go to next one
    x = int(boxes[i][0])
    y = int(boxes[i][1])
    w = int(boxes[i][2])
    h = int(boxes[i][3])
    
    cv2.rectangle(tf_image, (y, x), (h, w), random.choice(color)["rgb"], 2)

# Convert back to image for pillow and draw backgrounds for text
label_image = Image.fromarray(tf_image)
font = ImageFont.FreeTypeFont("AtariST8x16SystemFont.ttf", size=30)

# Write category and confidence score above box
# For every height index
for i in range(max_objects):
    # Go through each width
    for j in range(len(boxes[i])):
        x = int(boxes[i][0])
        y = int(boxes[i][1])
        w = int(boxes[i][2])
        h = int(boxes[i][3])
    string = " " + class_names[(int(class_ids[i])) - 1] + " | " + str(round(float(scores[i]), 3)) + " %"
    I1 = ImageDraw.Draw(label_image)
    I1.rounded_rectangle([y, x-32, y+260, x+4], fill=(30, 30, 30), radius=5)
    I1.text((y, x-30), string, font=font, fill=(235, 235, 235))

label_image
''')
# Image with tf labels on it
st.image("Obj_Det_TFHUB.jpg")

# Image Segmentation------------------------------------------------------------------------------

st.subheader("Image Segmentation with Tensorflow")
st.write("Image segmentation is the process of dividing an image into multiple segments or regions, each of which represents a different object or background. It is a crucial task in computer vision and has various applications, such as object recognition, autonomous driving, medical imaging, and more. The process of image segmentation involves detecting the boundaries of objects in an image and grouping the pixels that belong to the same object.")
st.write("One of the state-of-the-art models for image segmentation is the HRNet (High-Resolution Network), which is a deep convolutional neural network (CNN) architecture that has achieved high performance on various computer vision tasks, including image segmentation. HRNet has a multi-resolution fusion module that helps to extract features from different scales and resolutions, making it effective at capturing both local and global information in an image.")
st.write("The pre-trained HRNet model on TensorFlow Hub is trained on the COCO (Common Objects in Context) dataset, which contains more than 330,000 images and 2.5 million object instances, making it a robust model for image segmentation tasks. It is a variant of the HRNet architecture that has 48 weight layers and achieves high accuracy on the COCO dataset. This model can be used for a variety of image segmentation tasks, such as detecting objects in natural images, segmenting medical images, and more.")
st.subheader("CODE")
st.write("This example uses the same packages as above.")
st.code('''import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
''')
st.write("We will have to find the maximum confidence score for each pixel and return it. Define a function to detect our image segments. Pass the image and model and return predictions and features. Features will be added later.")
st.code('''# Create a mask of main object
def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def detect_seg(image, model):
    # Create a tensor out of the preprocessed image
    img = tf.cast(tf_preproc(image), tf.float32)/255.0
    # Predictions will have shape (batch_size, h, w, output_classes=134+1)
    # Note: an additional class is added for the background.
    predictions = model.predict(img)
    # Features will have shape (batch_size, h/4, w/4, 720)
    features = model.get_features(img)
    return predictions, features
''')
st.write("Using the load method, we setup the model. We are returned 202 predictions for each pixel in the H x W, so we get `predictions.shape = (756, 1008, 202)`")
st.code('''# Define the model
model = hub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')

# Get preds and features
predictions, features = detect_seg(original_image, model)
''')



st.write("Lets transform the predictions data into a few interesting things. First, create_mask gets the most confidents category and labels the pixel. Then, we need a numpy array instead of a tensor, so we wrap our create_mask call in `np.array()`.")

st.code('''# Create the array from mask and show image
pred_img_array = np.array(create_mask(predictions))
pred_img_array.shape
pred_img_array
''')
st.write("The shape after numpy conversion.")
st.code('''
    Output:
        (1, 756, 1008, 202)
''')
st.write("The image output.")
st.image("pred_img_array_mask.jpg")
st.write("The first few unique categories and their pixel counts.")
st.code('''# Visualize unique counts 
unique, counts = np.unique(pred_img_array, return_counts=True)
max_preds = 4
for num_preds in range(max_preds):
    print(unique[num_preds])
    print(counts[num_preds])
    print("-------------------")
''')
st.write("2 seems to be the winner, but let's make sure we have the right information!")
st.code('''
    Output:
        2
        430644
        -------------------
        10
        32602
        -------------------
        43
        2680
        -------------------
''')
st.write("Setting `main_obj` equal to `np.bincount(unique).argmax()` will ensure we choose the highest pixel count. We are choosing the largest detected object in the image, doing it this way. The people in this photo were detected as one object. Do you know of a technology that would detect them individually? Hint: Semantic vs Instance")
st.code('''# Find the obj with the highest count, Biggest on screen
main_obj = np.bincount(unique).argmax()
main_obj
''')
st.write("It was category 2. Can you find out what category that represents for this model?")
st.code('''
    Output:
        2
''')
st.write("Changing the shape to get just 2 axes.")
st.code('''# Squeeze
sq_arr = np.squeeze(pred_img_array)
sq_arr.shape
''')
st.code('''
    Output:
        (756, 1008)
''')
st.write("Here we loop through the array and set the value depending on if it was the la")
st.code('''# Cutout main object by change it to 255 and everything else to 0
sq_arr[sq_arr != main_obj] = 0
sq_arr[sq_arr == main_obj] = 255

sq_arr
''')
st.write("This is just the grayscale with the object being 255 and the rest 0. Let's use it to get transparency involved.")
st.image("cut_out_img.jpg")
st.write("We can use the array to build a RGBA image. Just stack the arrays and change the elements' type to uint8. Then `Image.fromarray()` will take it, and you can display a white cut-out of the object.")
st.code('''# Create a mask image with transparent backkground
rgba = np.dstack((sq_arr, sq_arr, sq_arr, sq_arr))

# Change type to uint8
cut_out = rgba.astype(np.uint8)

Image.fromarray(cut_out)
''')
st.image("cut_out_img_transparent_bg.png")
st.write("We can even add the channel array to the rgb to make a rgba, cut_out, of the original image.")
st.code('''# Create overlay
numpy_image_og = np.array(original_image)
overlay = np.dstack((numpy_image_og, sq_arr))

overlay
''')
st.write("How can we combine these computer vision methods to make better tools? Can you find any different/better image segmentation models than these? Hint: Yes!")
st.image("sample_cut_out.png")


