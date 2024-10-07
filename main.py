

import tkinter as tk
from tkinter import messagebox
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Italic', size=36, weight='bold')
window.title("Face Based Attendance Recognition")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
window.geometry("1000x600")
window.configure(bg="lightblue")

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)



def show_message(msg):
    messagebox.showinfo("Info", msg)

message = tk.Label(window, text="Face Based Attendance Recognition", bg="purple", fg="white", 
                   width=45, height=3, font=('times', 30, 'italic bold underline'))
message.place(x=150, y=20)

# ID Label
lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="white", bg="darkblue", 
               font=('times', 15, 'bold'))
lbl.place(x=350, y=200)

# ID Entry
txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, 'bold'))
txt.place(x=650, y=215)

# Name Label
lbl2 = tk.Label(window, text="Enter Name", width=20, fg="white", bg="darkblue", 
                height=2, font=('times', 15, 'bold'))
lbl2.place(x=350, y=275)

# Name Entry
txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, 'bold'))
txt2.place(x=650, y=290)


 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
          import unicodedata
          unicodedata.numeric(s)
          return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/+name" +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    show_message("Image Trained")


import os
import numpy as np
from PIL import Image

def getImagesAndLabels(path):
    # Get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    
    # Create empty face list
    faces = []
    # Create empty ID list
    Ids = []
    
    # Loop through all the image paths
    for imagePath in imagePaths:
        # Only process files with valid image extensions
        if imagePath.endswith((".jpg", ".jpeg", ".png")):
            try:
                # Load the image and convert it to grayscale
                pilImage = Image.open(imagePath).convert('L')
                # Convert the PIL image into a numpy array
                imageNp = np.array(pilImage, 'uint8')
                # Get the ID from the image filename
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                # Append the face and ID to respective lists
                faces.append(imageNp)
                Ids.append(Id)
            except Exception as e:
                print(f"Error processing file {imagePath}: {e}")
        else:
            print(f"Skipping non-image file: {imagePath}")
    
    return faces, Ids


import cv2
import os
import pandas as pd
import time
import datetime

def TrackImages():
    # Create LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")

    # Load Haarcascade for face detection
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Read student details CSV, skip bad lines
    df = pd.read_csv("StudentDetails/StudentDetails.csv", on_bad_lines='skip')

    # Initialize camera
    cam = cv2.VideoCapture(0)

    # Define font for displaying text on screen
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Columns for attendance dataframe
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        # Capture frame-by-frame
        ret, im = cam.read()

        # Convert to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Predict the face ID using the recognizer
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa[0]  # Name from CSV
                attendance.loc[len(attendance)] = [Id, aa[0], date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)

            # Save the image of unknown faces
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown/Image" + str(noOfFile) + ".jpg", im[y:y+h, x:x+w])

            # Display the name/ID on the frame
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)

        # Remove duplicate entries in attendance
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

        # Show the frame with rectangles and names
        cv2.imshow('im', im)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Save the attendance to a CSV file
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    Hour = datetime.datetime.fromtimestamp(ts).strftime('%H')
    Minute = datetime.datetime.fromtimestamp(ts).strftime('%M')
    Second = datetime.datetime.fromtimestamp(ts).strftime('%S')
    fileName = "Attendance/Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)

    # Release the camera and close windows
    cam.release()
    cv2.destroyAllWindows()

    # Display attendance result in message box or similar
    res = attendance
    show_message(res)


# Clear Buttons
clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="darkblue", 
                         width=15, height=1, activebackground="red", font=('times', 15, 'bold'))
clearButton.place(x=900, y=200)

clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="darkblue", 
                          width=15, height=1, activebackground="red", font=('times', 15, 'bold'))
clearButton2.place(x=900, y=280)

# Action Buttons
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="white", bg="orange", 
                    width=15, height=2, activebackground="red", font=('times', 15, 'bold'))
takeImg.place(x=350, y=550)

trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="white", bg="orange", 
                     width=15, height=2, activebackground="red", font=('times', 15, 'bold'))
trainImg.place(x=550, y=550)

trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="white", bg="orange", 
                     width=15, height=2, activebackground="red", font=('times', 15, 'bold'))
trackImg.place(x=750, y=550)

# Quit Button
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="darkblue", 
                       width=15, height=2, activebackground="red", font=('times', 15, 'bold'))
quitWindow.place(x=950, y=550)


 
window.mainloop()
time.sleep(10)