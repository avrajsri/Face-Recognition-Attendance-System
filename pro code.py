import os, cv2, csv, datetime
import numpy as np
import pandas as pd
from tkinter import *
from time import *
from PIL import Image

root = Tk()
root.geometry('690x550')
root.title("Face Recognition Attendance System")
root.resizable(width=FALSE, height=FALSE)

root.configure(background='#FFFFFF')

def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

center(root)
assure_path_exists("Attendance/")
assure_path_exists("TakeImages/")
assure_path_exists("TrainingImage/")
assure_path_exists("UnknownUser/")

def tick(time1=''):
    time2 = strftime('%H:%M:%S')
    if time2 != time1:
        l1.config(text=time2)
    l1.after(200, tick)

l1 = Label(root, font=('arial 15 bold'), background='#FFFFFF')
l1.place(x=370,y=10)
y = strftime("%d/%m/%y")
Label(root, text=y, font=('arial 16 bold'), background='#FFFFFF').place(x=250,y=10)
tick()

lbl = Label(root, text="Enter ID: ", width=20, fg="#0055ff", bg="white", font=('times', 17, ' bold '))
lbl.place(x=110, y=80)
txt = Entry(root, width=20, bg="white", fg="red",bd=4, font=('times', 15, ' bold '))
txt.focus_set()
txt.place(x=340, y=80)

lbl2 = Label(root, text="Enter Name: ", width=20, fg="#0055ff", bg="white", font=('times', 17, ' bold '))
lbl2.place(x=100, y=150)
txt2 = Entry(root, width=20, bg="white", fg="red",bd=4, font=('times', 15, ' bold '))
txt2.place(x=340, y=150)

lbl3 = Label(root, text="Notification: ", width=10, fg="red", bg="white", height=2, font=('times', 17, ' bold'))
lbl3.place(x=50, y=340)
message = Label(root, text="", fg="#0055ff", bg="white", relief=GROOVE, activebackground="yellow", width=35, height=2, font=('times', 15, ' bold '))
message.configure(text="Welcome Face Recognition Attendance System")
message.place(x=210, y=340)

lbl4 = Label(root, text="Attendance: ", width=10, fg="red", bg="white", height=2, font=('times', 17, ' bold'))
lbl4.place(x=50, y=445)

frame = Frame(root)
frame.place(x=210, y=420)
listNodes = Listbox(frame,  width=42, height=4, bd=4, font=('times', 15, ' bold '), selectbackground="blue", activestyle="none")
listNodes.pack(side="left", fill="y")
scrollbar = Scrollbar(frame, orient="vertical")
scrollbar.config(command=listNodes.yview)
scrollbar.pack(side="right", fill="y")
listNodes.config(yscrollcommand=scrollbar.set)

show="'Date'            'Time'             'Id'           'Name'                   "
lbl4 = Label(root, text=show, width=41, fg="#0055ff", bg="white", height=0, font=('times', 13, ' bold'))
lbl4.place(x=215, y=425)

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
    Id = (txt.get())
    name = (txt2.get())

    if(len(Id)==0 & len(name)==0):
        res = "Enter Numeric Id"
        message.configure(text=res)

    if (is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                sampleNum = sampleNum + 1
                cv2.imwrite("TakeImages\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
            if cv2.waitKey(25) & 0xFF == ord('a'):
                break
            elif sampleNum > 30:
                break

        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('Details\Details.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)


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

    faces, Id = getImagesAndLabels("TakeImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImage\Trainner.yml")

    message.configure(text="Image Trained")

def TrackImages():
    listNodes.delete(0,END)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImage\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)

    df = pd.read_csv("Details\Details.csv")
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

            if (conf < 50):
                ts = time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = df.loc[df['Id'] == Id]['Name'].values
                name=(name[0])
                mess = str(Id) + " " + name
                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

            else:
                Id = 'Unknown'
                mess = str(Id)

            if (conf > 75):
                noOfFile = len(os.listdir("UnknownUser")) + 1
                cv2.imwrite("UnknownUser\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])

            cv2.putText(im, str(mess), (x, y - 40), font, 1, (255, 255, 255), 3)

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Track Images', im)
        if cv2.waitKey(10) & 0xFF == ord('a'):
            break

    ts = time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")

    fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)

    cam.release()
    cv2.destroyAllWindows()
    message.configure(text="Attendance Complete")

    with open(fileName, 'r')as r:
        data = csv.reader(r)
        z1="   "
        z2="     "
        z3="     "
        z4="          "
        for row in data:
            listNodes.insert(END, z1 + row[2] + z2 + row[3] + z3 + row[0] + z4 + row[1])


takeImg = Button(root, text="Take Images", command=TakeImages, bg="#3eff00", activebackground="#0055ff", activeforeground="white", bd=4, font=('times', 15, ' bold '))
takeImg.place(x=50, y=240)
trainImg = Button(root, text="Train Images", command=TrainImages, bg="#3eff00", activebackground="#0055ff", activeforeground="white", bd=4, font=('times', 15, ' bold '))
trainImg.place(x=220, y=240)
trackImg = Button(root, text="Track Images", command=TrackImages, bg="#3eff00", activebackground="#0055ff", activeforeground="white", bd=4, font=('times', 15, ' bold '))
trackImg.place(x=390, y=240)
quitroot = Button(root, text="Quit",fg="white", command=root.destroy, bg="#ff1111", activebackground="Red", activeforeground="white", bd=4, font=('times', 15, ' bold '))
quitroot.place(x=570, y=240)
root.mainloop()
