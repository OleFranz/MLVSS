import SimpleWindow
import ImageUI
import ctypes
import shutil
import mouse
import numpy
import time
import cv2
import os


AnnotationValue = 0
Path = os.path.dirname(os.path.dirname(__file__)).replace("\\", "/")
if Path[-1] != "/": Path += "/"
SourcePath = Path + "AdaptiveCruiseControl/Datasets/RawDataset/"
DestinationPath = Path + "AdaptiveCruiseControl/Datasets/AnnotatedDataset/"
Background = numpy.zeros((900, 500, 3), numpy.uint8)

os.makedirs(SourcePath, exist_ok=True)
os.makedirs(DestinationPath, exist_ok=True)

SimpleWindow.Initialize(Name="Annotation",
                        Size=(500, 900),
                        Position=(100, 100),
                        TitleBarColor=(47, 47, 47),
                        Resizable=False,
                        TopMost=False,
                        Undestroyable=False)
SimpleWindow.Show("Annotation", Background)

Index = 0
Files = [File for File in os.listdir(SourcePath) if File.endswith(".png")]
if len(Files) == 0:
    print("Dataset is empty!")
    exit()

def SetIndex(IndexValue):
    global Index
    Index = IndexValue

def SetRunLoop(RunLoopValue):
    global RunLoop
    RunLoop = RunLoopValue

def Save():
    with open(DestinationPath + Files[Index].replace(".png", ".txt"), "w") as File:
        File.write(str(AnnotationValue))
    shutil.copy2(SourcePath + Files[Index], DestinationPath + Files[Index])

while True:
    File = Files[Index]

    ImageOriginal = cv2.imread(SourcePath + File)
    ImageOriginal = cv2.resize(ImageOriginal, (Background.shape[1], Background.shape[0]))

    if os.path.exists(DestinationPath + File.replace(".png", ".txt")):
        with open(DestinationPath + File.replace(".png", ".txt"), "r") as File:
            AnnotationValue = float(File.read().strip())

    RunLoop = True
    while RunLoop:
        if SimpleWindow.GetOpen("Annotation") != True: break
        Start = time.time()
        Frame = Background.copy()
        Image = ImageOriginal.copy()
        MouseX, MouseY = mouse.get_position()
        WindowX, WindowY = SimpleWindow.GetPosition("Annotation")
        XoverWindow = MouseX - WindowX
        YoverWindow = MouseY - WindowY
        if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and SimpleWindow.GetForeground("Annotation") and 0 <= XoverWindow < Background.shape[1] - 1 and -20 <= YoverWindow < Background.shape[0] - 50:
            AnnotationValue = min(max(0, YoverWindow / (Background.shape[0] - 50)), 1)
        cv2.line(Image, (0, round(AnnotationValue * Background.shape[0])), (Background.shape[1], round(AnnotationValue * Background.shape[0])), (0, 255, 255), 2)

        ImageUI.Image(Image=Image,
                      X1=0,
                      Y1=0,
                      X2=Background.shape[1] - 1,
                      Y2=Background.shape[0] - 51,
                      ID="AnnotationImage")

        ImageUI.Button(Text="Back",
                       X1=5,
                       Y1=Background.shape[0] - 45,
                       X2=Background.shape[1] / 3 - 3,
                       Y2=Background.shape[0] - 6,
                       ID="Back",
                       OnPress=lambda: {Save(), SetIndex(Index - (1 if Index > 0 else 0)), SetRunLoop(False)})

        ImageUI.Button(Text="Save",
                       X1=Background.shape[1] / 3 + 3,
                       Y1=Background.shape[0] - 45,
                       X2=Background.shape[1] / 1.5 - 3,
                       Y2=Background.shape[0] - 6,
                       ID="Save",
                       OnPress=lambda: Save())

        ImageUI.Button(Text="Next",
                       X1=Background.shape[1] / 1.5 + 3,
                       Y1=Background.shape[0] - 45,
                       X2=Background.shape[1] - 6,
                       Y2=Background.shape[0] - 6,
                       ID="Next",
                       OnPress=lambda: {Save(), SetIndex(Index + (1 if Index < len(Files) - 1 else 0)), SetRunLoop(False)})
    
        Frame = ImageUI.Update(WindowHWND=SimpleWindow.GetHandle("Annotation"), Frame=Frame)
        SimpleWindow.Show("Annotation", Frame)

        TimeToSleep = 1/60 - (time.time() - Start)
        if TimeToSleep > 0:
            time.sleep(TimeToSleep)

    if SimpleWindow.GetOpen("Annotation") != True: break