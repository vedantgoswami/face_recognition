#importing the necessary module
import os
import cv2
import face_recognition
#passing the file location of known and unknown faces\
KNOWN_FACES_DIR = r'F:\face_recognition\Known_faces'
UNKNOWN_FACES_DIR = r'F:\face_recognition\Unknown_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 5
FONT_THICKNESS = 4
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
print('Loading known faces...')
known_faces = []
known_names = []
# NOTE: known faces directory can have more than one person face but please keep them in a seperate file and name the with their name.

for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        print("Loading image:-",filename)
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        
        known_faces.append(encoding)
        known_names.append(name)
print("Done!..........")


print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    # location contains the end points of faces 
    locations = face_recognition.face_locations(image, model=MODEL)

    # now we will encode the face whication have the above locations because encoding the whole image is time consuming and not efficient
    
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = (50,205,50)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 30)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (84, 84, 252), FONT_THICKNESS)
    image = cv2.resize(image, (1000,1000)) 
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)
