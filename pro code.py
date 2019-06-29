import os,cv2,csv,time,datetime
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image

window = tk.Tk()
window.geometry('700x530')
window.title("Face-Recognition-Based-Attendance-System-master")
lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="red", bg="yellow", font=('times', 15, ' bold '))
lbl.place(x=100, y=50)
txt = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt.place(x=400, y=50)
lbl2 = tk.Label(window, text="Enter Name", width=20, fg="red", bg="yellow", height=2, font=('times', 15, ' bold '))
lbl2.place(x=100, y=150)
txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt2.place(x=400, y=150)
message = tk.Label(window, text="", bg="yellow", fg="red", width=35, height=2, activebackground="yellow",
                   font=('times', 15, ' bold '))
message.place(x=220, y=350)
lbl3 = tk.Label(window, text="Attendance: ", width=10, fg="red", bg="yellow", height=2, font=('times', 15, ' bold'))
lbl3.place(x=100, y=450)
message2 = tk.Label(window, text="", fg="red", bg="yellow", activeforeground="green", width=30, height=3,
                    font=('times', 15, ' bold '))
message2.place(x=280, y=440)


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
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)
            if sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
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
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")

    message.configure(text="Image Trained")

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)

    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)

            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])

            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)

        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im)
        if cv2.waitKey(50) & 0xFF == ord('a'):
            break

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)

    cam.release()
    cv2.destroyAllWindows()
    message2.configure(text= attendance)

  
takeImg = tk.Button(window, text="Take Images", command=TakeImages, bg="yellow", activebackground="Red",
                    font=('times', 15, ' bold '))
takeImg.place(x=100, y=250)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, bg="yellow", activebackground="Red",
                     font=('times', 15, ' bold '))
trainImg.place(x=270, y=250)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, bg="yellow", activebackground="Red",
                     font=('times', 15, ' bold '))
trackImg.place(x=450, y=250)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, bg="yellow", activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=100, y=350)
window.mainloop()
