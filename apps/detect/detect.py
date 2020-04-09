# import the necessary packages
import appdaemon.plugins.hass.hassapi as hass
from __future__ import print_function
#from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import os
from pathlib import Path
import cv2
import fnmatch



class Detect(hass.Hass):

    def initialize(self):
        self.actions = self.args.get("actions", [])

        self.perform_action( )        


    def perform_action
        if type(self.buttons) is not list:
            self.buttons = [self.buttons]

        for button in self.buttons:
            self.listen_event(self.cb_button_press, "xiaomi_aqara.click",
                              entity_id = button)


        DEFAULT_MIN_SIZE = (30, 30)
        DEFAULT_NEIGHBORS = 4
        DEFAULT_SCALE = 1.1
        # construct the argument parse and parse the arguments
        #ap = argparse.ArgumentParser()
        #ap.add_argument("-i", "--images", required=True,
        #                help="path to images directory")
        #args = vars(ap.parse_args())
        size = 950
        upper_model = '/config/opencv/haarcascade_upperbody.xml'
        face_model = '/config/opencv/haarcascade_upperbody.xml'
        dest_folder = '/share/droit/to/' 
        source_folder = '/share/droit/' 
        
        #rename existing files
        current_files = sorted(fnmatch.filter(os.listdir(dest_folder), '*image*.jpg'))
        for file_old in current_files:
        #    os.rename( dest_folder + file_old, dest_folder + 'old_' + file_old)
            os.remove(dest_folder+file_old)
        	#print(file_old)
        
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        i = 0
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(face_model)
        upper_cascade = cv2.CascadeClassifier(upper_model)
        
        # loop over the image paths
        current_files = sorted(fnmatch.filter(os.listdir(source_folder), '*.jpg'))
        #print("nombre de fichiers:",len(current_files))
        for file_old in current_files:
        
        #for imagePath in sorted(paths.list_images(args["images"])):
        
          # load the image and resize it to (1) reduce detection time
            # and (2) improve detection accuracy
        #    image = cv2.imread(imagePath)
            image = cv2.imread(source_folder+file_old)
            image = imutils.resize(image, width=min(size, image.shape[1]))
            orig = image.copy()     
            # detect people in the image
            #(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
            #(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)
        
            # detect people in the image
            #image2 = cv2.imread(imagePath)
            #image2 = imutils.resize(image, width=min(400, image.shape[1]))
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        #    upper = upper_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
        #    imagecrop = image2[int((size//5)*3):10,int(size - (size // 20)):200]
        
        #right part
            height, width = image.shape[0:2]
            startRow = int(height*.28)
            startCol = int(width*.55)
            endRow = int(height*.85)
            endCol = int(width*.92)
            croppedImage = image[startRow:endRow, startCol:endCol]
            gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
            #(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)    
            (rects, weights) = hog.detectMultiScale(croppedImage , winStride=(8, 8), padding=(16, 16), scale=1.07)    
        	#(rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)    
        #    faces = face_cascade.detectMultiScale(image, scaleFactor=DEFAULT_SCALE, minNeighbors=DEFAULT_NEIGHBORS, minSize=DEFAULT_MIN_SIZE)
        #    upper = upper_cascade.detectMultiScale(image, scaleFactor=DEFAULT_SCALE, minNeighbors=DEFAULT_NEIGHBORS, minSize=DEFAULT_MIN_SIZE)
            upper = upper_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=7)    
          
          
        ## bon résultats
            #(rects, weights) = hog.detectMultiScale(croppedImage , winStride=(8, 8), padding=(16, 16), scale=1.08)    
            #upper = upper_cascade.detectMultiScale(croppedImage, scaleFactor=1.01, minNeighbors=8)    
        
          
          
          
            cv2.rectangle(image, ( startCol, startRow ), (endCol, endRow), (0, 255, 0), 2)  
            
            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(image, (x+startCol, y+startRow), (x + w+startCol, y + h+startRow), (255, 0, 0), 2)
        
        
            # draw the original bounding boxes in red
            for (x, y, w, h) in upper:
                cv2.rectangle(image, (x+startCol, y+startRow), (x + w+startCol, y + h+startRow), (0, 0, 255), 2)
        
          
        #left part        
            #image = imutils.resize(image, width=min(100, image.shape[1]))
            #height, width = image.shape[0:2]
            startRow = int(height*.30)
            startCol = int(width*.20)
        
            endRow = int(height*.99)
            endCol = int(width*.6)
            croppedImage = image[startRow:endRow, startCol:endCol]
            gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)       
            cv2.rectangle(image, ( startCol, startRow ), (endCol, endRow), (255, 0, 0), 2)  
            
            upper2 = upper_cascade.detectMultiScale(croppedImage, scaleFactor=1.11, minNeighbors=7)    
            (rects2, weights) = hog.detectMultiScale(croppedImage , winStride=(8, 8), padding=(32, 32), scale=1.04)            
            faces = face_cascade.detectMultiScale(croppedImage, scaleFactor=1.4, minNeighbors=4)
                
                
            # draw the original bounding boxes in red
            for (x, y, w, h) in upper2:
                cv2.rectangle(image, (x+startCol, y+startRow), (x + w+startCol, y + h+startRow), (0, 0, 255), 2)
        
            #blue?
            for (x, y, w, h) in rects2:
                cv2.rectangle(image, (x+startCol, y+startRow), (x + w+startCol, y + h+startRow), ( 255, 0, 0), 2)
        
                    # draw the original bounding boxes #green
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x+startCol, y+startRow), (x + w+startCol, y + h+startRow), (0, 255, 0), 2)        
        
                
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            #pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        
            # draw the final bounding boxes black
            #for (xA, yA, xB, yB) in pick:
            #for (xA, yA, xB, yB) in rects:      
            #    cv2.rectangle(image, (xA, yA), (xB, yB), (10, 10, 10), 2)
        
            # show some information on the number of bounding boxes
            #filename = imagePath[imagePath.rfind("/") + 1:]
            #filename = dest_folder + '/' + str(i).zfill(2) + 'image' + '.jpg'
            filename = dest_folder + file_old
        
            
            #sauvegarde image
            if len(faces) + len(rects) + + len(rects2) + len(upper) + len(upper2) > 0:
                self.log("detection)
             #   print("faces:",len(faces),"/HOG:", len(rects), "/", len(rects2) ,"/upperbody",len(upper),"/",len(upper2))
          #  if 1 == 1:
          #if len(faces) + len(upper) > 0:
        
            #    #renomme les fichiers en +1
            #    for x in range(30, 20):
            #      fromfile = imagepath + str(19-x) + '.jpg'
            #      tofile = imagepath + str(20-x) + '.jpg'
            #      src = Path(fromfile)
            #      dest = Path(tofile)
            #      if src.exists():
            #          if dest.exists():
            #              os.remove(tofile)  
            #          print('rename '+fromfile+'to '+tofile)
            #          os.rename(fromfile, tofile)
                #filename = dest_folder  + str(i).zfill(2) + 'image' + '.jpg'
                filename = dest_folder + file_old
                #        os.remove(filename)
                cv2.imwrite(filename, image)
                #cv2.imwrite(filename, orig)        
                i = i + 1
           #     print("detection personne  : ", file_old, "n°:", i)
                #print("[INFO] {}: {} original boxes, {} after suppression _ {} Upperbody".format(
                #    filename, len(rects), len(pick), len(faces)))
            #if len(faces) < -1 and len(rects) < 1 and len(upper) < 1:
            else:
                filename = dest_folder + file_old
                #        os.remove(filename)
                cv2.imwrite(filename, image)
                  
            if len(faces) < -1 and len(upper) < 1:
            #    print("vide")
            #    os.remove(imagePath)
        #    os.remove(imagePath)
        #rename old to  new
        #for files_new in current_files:
        #    if i < 30:
        #        #print(dest_folder + 'image' + str(i) + '.jpg')
        #        os.rename(dest_folder + 'old_' + files_new, dest_folder + str(i).zfill(2) + 'image'  + '.jpg') 
        #    else:
        #        #print(files_new)
        #        os.remove(dest_folder + 'old_' + files_new)
        #    i = i + 1
            # show the output images
            #cv2.imshow("Before NMS", orig)
            #cv2.imshow("After NMS", image)
            # cv2.waitKey(0)
