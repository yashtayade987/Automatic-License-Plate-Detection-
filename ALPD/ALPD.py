import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

# Import image
img = cv2.imread('test.png')

# Grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display grayscale image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Image')
plt.axis('on')
plt.show()

# Noise reduction
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Display edged image
plt.imshow(edged, cmap='gray')
plt.title('Edged Image')
plt.axis('on')
plt.show()

# Path to Haar cascade
harcascade = "C:/Users/yasht/Desktop/python/lisance plate detection/model/haarcascade_russian_plate_number.xml"
min_area = 500
count = 0

# Initialize the Haar cascade classifier
plate_cascade = cv2.CascadeClassifier(harcascade)
plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

# Initialize easyocr reader
reader = easyocr.Reader(['en'])

for (x, y, w, h) in plates:
    area = w * h

    if area > min_area:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        img_roi = img[y: y + h, x: x + w]
        cv2.imshow("ROI", img_roi)

        # Save the detected plate region
        save_path = f"C:/Users/yasht/Desktop/python/lisance plate detection/plates/scanned_img_{count}.jpg"
        cv2.imwrite(save_path, img_roi)

        # Use easyocr to read text from the image ROI
        ocr_result = reader.readtext(img_roi)
        
        # Extract and print the detected text
        for result in ocr_result:
            text = result[1]
            print(f"Detected Number Plate Text: {text}")

# Display the result for the current image
cv2.imshow("Result", img)
cv2.waitKey(0)  # Wait for a key press to move to the next image

# Release resources
cv2.destroyAllWindows()
