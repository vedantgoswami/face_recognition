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
and the cormer location of every face is stored in the location variable and only that area is encoded 

### Here are the face detection sample 
![Capture1](https://user-images.githubusercontent.com/56600948/84247186-55368f00-ab25-11ea-9b8c-3f55e93f7b87.PNG)
