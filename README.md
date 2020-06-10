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
![Capture1](https://user-images.githubusercontent.com/56600948/84248081-8e233380-ab26-11ea-9974-d1d484dc88bf.PNG)
![Capture](https://user-images.githubusercontent.com/56600948/84248091-91b6ba80-ab26-11ea-9492-65a261ffc57a.PNG)
![Capture2](https://user-images.githubusercontent.com/56600948/84248105-967b6e80-ab26-11ea-8049-f285c428f449.PNG)

