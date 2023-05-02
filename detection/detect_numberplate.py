# import required libraries
import cv2
import numpy as np
import aiofiles
from fastapi import UploadFile 
import pytesseract 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read input image
class LazyLoader:
   def __init__(self):
      self.cascade = None
   def load_model(self):
      if self.cascade == None:
         self.cascade = cv2.CascadeClassifier(r'C:\Users\Shraddha\OneDrive\Documents\Desktop\Projects\NoPlateDetection\detection\haarcascade_russian_plate_number.xml')
         return self.cascade
      else: 
         return self.cascade

lazy_loader_obj = LazyLoader()

async def save_file(file, path):
   async with aiofiles.open(f'{path}/{file.filename}', 'wb') as downloaded_file:
      content = await file.read()
      await downloaded_file.write(content)

async def detect(image: UploadFile): 

   path = "download"
   await save_file(image, path)
   # Load the model
   cascade = lazy_loader_obj.load_model()
   img = cv2.imread(f'{path}/{image.filename}')

   # convert input image to grayscale
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # read haarcascade for number plate detection
   # cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

   # Detect license number plates
   plates = cascade.detectMultiScale(gray, 1.2, 5)
   print('Number of detected license plates:', len(plates))

   # loop over all plates
   numlist = []
   for (x,y,w,h) in plates:
      
      # draw bounding rectangle around the license number plate
      cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
      gray_plates = gray[y:y+h, x:x+w]
      color_plates = img[y:y+h, x:x+w]
      
      # save number plate detected
      cv2.imwrite('images/Numberplate.jpg', gray_plates)
      text = pytesseract.image_to_string('images/Numberplate.jpg')
      print("text",text)
      numlist.append(text)


   #    cv2.imshow('Number Plate', gray_plates)
   #    cv2.imshow('Number Plate Image', img)
   #    cv2.waitKey(0)
   # cv2.destroyAllWindows()
   return numlist

#detection("hundai.jpg")
if __name__ == "__main__":
   detect("mc.jpg")