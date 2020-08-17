import streamlit as st 
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import cv2
#from PIL import Image,ImageEnhance
from sudokuimage import find_puzzle,extract_digit
from sudoku import Sudoku 

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
model = load_model("digit_classifier.h5")

WIDTH = 320

#image = Image.open("titleimage.jpg")
st.image("titleimage.jpg", use_column_width=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

activities = ["Sudoku", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == 'Sudoku':
	st.title("Sudoku solver")

	uploaded_image = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
	if uploaded_image is not None:
		#image = Image.open(image_file)
		#image = np.array(image)
		image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
		st.sidebar.info("Original Image")
		st.sidebar.image(image,width=WIDTH)


		showprocess = st.checkbox("show sudoku extract process")
		if showprocess :
			(puzzleImage, warped) = find_puzzle(image, debug=True)
		else :
			(puzzleImage, warped) = find_puzzle(image, debug=False)
		st.image(puzzleImage,width=WIDTH)


		showprediction = st.checkbox("show digit prediction")
		# initialize our 9x9 sudoku board
		board = np.zeros((9, 9), dtype="int")

		# a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
		# infer the location of each cell by dividing the warped image
		# into a 9x9 grid
		stepX = warped.shape[1] // 9
		stepY = warped.shape[0] // 9

		# initialize a list to store the (x, y)-coordinates of each cell
		# location
		cellLocs = []

		# loop over the grid locations
		for y in range(0, 9):
			# initialize the current list of cell locations
			row = []

			for x in range(0, 9):
				# compute the starting and ending (x, y)-coordinates of the
				# current cell
				startX = x * stepX
				startY = y * stepY
				endX = (x + 1) * stepX
				endY = (y + 1) * stepY

				# add the (x, y)-coordinates to our cell locations list
				row.append((startX, startY, endX, endY))

				# crop the cell from the warped transform image and then
				# extract the digit from the cell
				cell = warped[startY:endY, startX:endX]
				digit = extract_digit(cell, debug=False)

				# verify that the digit is not empty
				if digit is not None:
					foo = np.hstack([cell, digit])
					#cv2.imshow("Cell/Digit", foo)
					

					# resize the cell to 28x28 pixels and then prepare the
					# cell for classification
					roi = cv2.resize(digit, (28, 28))
					roi = roi.astype("float") / 255.0
					roi = img_to_array(roi)
					roi = np.expand_dims(roi, axis=0)

					# classify the digit and update the sudoku board with the
					# prediction
					pred = model.predict(roi).argmax(axis=1)[0]
					board[y, x] = pred
					if showprediction :
						st.image(foo)
						st.write(pred)

			# add the row to our cell locations
			cellLocs.append(row)

		# construct a sudoku puzzle from the board
		print("[INFO] OCR'd sudoku board:")
		puzzle = Sudoku(3, 3, board=board.tolist())
		puzzle.show()

		# solve the sudoku puzzle
		print("[INFO] solving sudoku puzzle...")
		solution = puzzle.solve()
		solution.show_full()



		# loop over the cell locations and board
		for (cellRow, boardRow) in zip(cellLocs, solution.board):
			# loop over individual cell in the row
			for (box, digit) in zip(cellRow, boardRow):
				# unpack the cell coordinates
				startX, startY, endX, endY = box

				# compute the coordinates of where the digit will be drawn
				# on the output puzzle image
				textX = int((endX - startX) * 0.33)
				textY = int((endY - startY) * -0.2)
				textX += startX
				textY += endY

				# draw the result digit on the sudoku puzzle image
				cv2.putText(puzzleImage, str(digit), (textX, textY),
		              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

		# show the output image
		if st.button("solve"):
			st.image(puzzleImage,width=WIDTH)
	
elif choice == 'About':
	st.title("About")
	st.write("Using streamlit , mnist digit recognition model and py-sudoku package created a simple sudoku solver web app")
	st.text("Author:Jammula Rakesh Kumar")
	st.text("Credit:Pyimagesearch")
