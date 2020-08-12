import streamlit as st 
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
from PIL import Image,ImageEnhance

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


st.title("Sudoku solver")
st.set_option('deprecation.showfileUploaderEncoding', False)
image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
	image = Image.open(image_file)
	image = np.array(image)
	st.text("Original Image")
	st.image(image,width=WIDTH)
