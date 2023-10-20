import cv2

facePath = "src\\face.xml"
eyeglassPath = "src\\eyeglass.xml"
smilePath = "src\\smile.xml"

faceCascade = cv2.CascadeClassifier(facePath)
eyeglassCascade = cv2.CascadeClassifier(eyeglassPath)
smileCascade = cv2. CascadeClassifier(smilePath)


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    eyeglass = eyeglassCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    smile = smileCascade.detectMultiScale(
        gray,
        scaleFactor=1.8,
        minNeighbors=20,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
   # Draw rectangles around the faces
    for (x, y, w, h) in face:        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)        

    # Draw rectangles around the eyeglasses
    for (x, y, w, h) in eyeglass:        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 100), 2)
    
    # Draw rectangles around the smiles
    for (x, y, w, h) in smile:        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 0, 100), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()