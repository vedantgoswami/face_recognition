# face_recognition
## Introduction

This project is the simple **Face_Recognition** Project with Open CV module.
For this you need cv2 module. You can use below code to install it.
> pip install opencv

**It basically involves two steps:**
* Face detection
* Face recognition

### Face detection
~~~
image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
locations = face_recognition.face_locations(image, model=MODEL)
encodings = face_recognition.face_encodings(image, locations)
~~~

In the above code images present in the unknown directory are loaded and the all the faces in the images are detected.
and the corner point location of every face is stored in the location variable and only that area is encoded 

### Here are the face detection sample 
![Capture1](https://user-images.githubusercontent.com/56600948/84248081-8e233380-ab26-11ea-9974-d1d484dc88bf.PNG)
![Capture](https://user-images.githubusercontent.com/56600948/84248091-91b6ba80-ab26-11ea-9492-65a261ffc57a.PNG)
![Capture2](https://user-images.githubusercontent.com/56600948/84248105-967b6e80-ab26-11ea-8049-f285c428f449.PNG)

### Face recognition
After this the encoded faces are compared with the known faces and if the faces match then we assign the label (which is actually the folder name in which the images of known person is kept)

> results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

_NOTE_ :**Tolerance** – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
                        you can achive higher accuracy by decreasing the threshold
                        
  ### Here is the detected face of mine
![Capture4](https://user-images.githubusercontent.com/56600948/84250547-ced07c00-ab29-11ea-819b-09e4792a5f5e.png)
![Capture3](https://user-images.githubusercontent.com/56600948/84250558-d2fc9980-ab29-11ea-871e-60982fb3f8b8.png)
![Capture4](https://user-images.githubusercontent.com/56600948/84252135-d4c75c80-ab2b-11ea-8d13-1c148d329068.PNG)




## Thank You 

                        
                        
