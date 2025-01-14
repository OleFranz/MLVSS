from ctypes import windll, byref, sizeof, c_int
import numpy as np
import win32con
import win32gui
import shutil
import ctypes
import mouse
import math
import time
import cv2
import os

PATH = str(os.path.dirname(os.path.dirname(__file__))).replace("\\", "/")
if PATH[-1] != "/": PATH += "/"
DATA_PATH = PATH + "Mapping/Datasets/RawDataset/"
DST_PATH = PATH + "Mapping/Datasets/AnnotatedDataset/"

if os.path.exists(DST_PATH) == False:
    os.makedirs(DST_PATH)
if os.path.exists(DATA_PATH) == False:
    os.makedirs(DATA_PATH)

FPS = 60

Index = 0
LastLeftClicked = False
LastRightClicked = False
LastWindowSize = None, None
GrabbedPoint = None, None, None

print("Caching Images...")

Images = []
for File in os.listdir(DATA_PATH):
    if (File.endswith(".png") or File.endswith(".jpg") or File.endswith(".jpeg")) and os.path.exists(f"{DST_PATH}{File.replace(File.split('.')[-1], 'txt')}") == False:
        Image = cv2.imread(f"{DATA_PATH}{File}", cv2.IMREAD_UNCHANGED)
        Images.append((Image, File))

print("Done!")

def CreateWindow():
    cv2.namedWindow("Road Segmentation - Annotation", cv2.WINDOW_NORMAL)
    if os.name == 'nt':
        HWND = win32gui.FindWindow(None, "Road Segmentation - Annotation")
        windll.dwmapi.DwmSetWindowAttribute(HWND, 35, byref(c_int(0x000000)), sizeof(c_int))
        IconFlags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        HICON = win32gui.LoadImage(None, f"{PATH}icon.ico", win32con.IMAGE_ICON, 0, 0, IconFlags)
        win32gui.SendMessage(HWND, win32con.WM_SETICON, win32con.ICON_SMALL, HICON)
        win32gui.SendMessage(HWND, win32con.WM_SETICON, win32con.ICON_BIG, HICON)
CreateWindow()

def GetTextSize(Text="NONE", TextWidth=100, MaxTextHeight=100):
    Fontscale = 1
    Textsize, _ = cv2.getTextSize(Text, cv2.FONT_HERSHEY_SIMPLEX, Fontscale, 1)
    WidthCurrentText, HeightCurrentText = Textsize
    MaxCountCurrentText = 3
    while WidthCurrentText != TextWidth or HeightCurrentText > MaxTextHeight:
        Fontscale *= min(TextWidth / Textsize[0], MaxTextHeight / Textsize[1])
        Textsize, _ = cv2.getTextSize(Text, cv2.FONT_HERSHEY_SIMPLEX, Fontscale, 1)
        MaxCountCurrentText -= 1
        if MaxCountCurrentText <= 0:
            break
    Thickness = round(Fontscale * 2)
    if Thickness <= 0:
        Thickness = 1
    return Text, Fontscale, Thickness, Textsize[0], Textsize[1]


def Button(Text="NONE", X1=0, Y1=0, X2=100, Y2=100, RoundCorners=30, Buttoncolor=(100, 100, 100), Buttonhovercolor=(130, 130, 130), Buttonselectedcolor=(160, 160, 160), Buttonselectedhovercolor=(190, 190, 190), Buttonselected=False, Textcolor=(255, 255, 255), WidthScale=0.9, HeightScale=0.8):
    if X1 <= MouseX*FrameWidth <= X2 and Y1 <= MouseY*FrameHeight <= Y2:
        ButtonHovered = True
    else:
        ButtonHovered = False
    if Buttonselected == True:
        if ButtonHovered == True:
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonselectedhovercolor, RoundCorners, cv2.LINE_AA)
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonselectedhovercolor, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonselectedcolor, RoundCorners, cv2.LINE_AA)
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonselectedcolor, -1, cv2.LINE_AA)
    elif ButtonHovered == True:
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonhovercolor, RoundCorners, cv2.LINE_AA)
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttonhovercolor, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttoncolor, RoundCorners, cv2.LINE_AA)
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), Buttoncolor, -1, cv2.LINE_AA)
    Text, Fontscale, Thickness, Width, Height = GetTextSize(Text, round((X2-X1)*WidthScale), round((Y2-Y1)*HeightScale))
    cv2.putText(Frame, Text, (round(X1 + (X2-X1) / 2 - Width / 2), round(Y1 + (Y2-Y1) / 2 + Height / 2)), cv2.FONT_HERSHEY_SIMPLEX, Fontscale, Textcolor, Thickness, cv2.LINE_AA)
    if X1 <= MouseX*FrameWidth <= X2 and Y1 <= MouseY*FrameHeight <= Y2 and Leftclicked == False and LastLeftclicked == True:
        return True, ButtonHovered
    else:
        return False, ButtonHovered

while Index < len(Images):
    Start = time.time()

    try:
        WindowX, WindowY, WindowWidth, WindowHeight = cv2.getWindowImageRect("Road Segmentation - Annotation")
        if WindowWidth < 50 or WindowHeight < 50:
            cv2.resizeWindow("Road Segmentation - Annotation", 50, 50)
            WindowWidth = 50
            WindowHeight = 50
        if WindowWidth != LastWindowSize[0] or WindowHeight != LastWindowSize[1]:
            LastWindowSize = WindowWidth, WindowHeight
            Background = np.zeros((WindowHeight, WindowWidth, 3), np.uint8)
        MouseX, MouseY = mouse.get_position()
        MouseRelativeWindow = MouseX - WindowX, MouseY - WindowY
        if WindowWidth != 0 and WindowHeight != 0:
            MouseX = MouseRelativeWindow[0]/WindowWidth
            MouseY = MouseRelativeWindow[1]/WindowHeight
        else:
            MouseX = 0
            MouseY = 0
        MouseXImage = MouseX / 0.7
        MouseYImage = MouseY
    except:
        exit()

    if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "Road Segmentation - Annotation"):
        Leftclicked = True
    else:
        Leftclicked = False

    if ctypes.windll.user32.GetKeyState(0x02) & 0x8000 != 0 and ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "Road Segmentation - Annotation"):
        RightClicked = True
    else:
        RightClicked = False

    Frame = Background.copy()
    FrameHeight, FrameWidth, _ = Frame.shape
    Image = cv2.resize(Images[Index][0], (round(Background.shape[1] * 0.7), Background.shape[0]))

    ButtonNextPressed, ButtonNextHovered = Button(Text="Next",
                                                      X1=0.85125*FrameWidth,
                                                      Y1=0.005*FrameHeight,
                                                      X2=0.9975*FrameWidth,
                                                      Y2=0.245*FrameHeight,
                                                      RoundCorners=30,
                                                      Buttoncolor=(0, 200, 0),
                                                      Buttonhovercolor=(20, 220, 20),
                                                      Buttonselectedcolor=(20, 220, 20),
                                                      Textcolor=(255, 255, 255),
                                                      WidthScale=0.95,
                                                      HeightScale=0.5)

    ButtonBackPressed, ButtonBackHovered = Button(Text="Back",
                                                      X1=0.7025*FrameWidth,
                                                      Y1=0.005*FrameHeight,
                                                      X2=0.84825*FrameWidth,
                                                      Y2=0.245*FrameHeight,
                                                      RoundCorners=30,
                                                      Buttoncolor=(0, 0, 200),
                                                      Buttonhovercolor=(20, 20, 220),
                                                      Buttonselectedcolor=(20, 20, 220),
                                                      Textcolor=(255, 255, 255),
                                                      WidthScale=0.95,
                                                      HeightScale=0.5)

    if ButtonNextPressed == True and Index < len(Images) - 1:
        try:
            shutil.copy2(f"{DATA_PATH}{Images[Index][1]}", f"{DST_PATH}{Images[Index][1]}")
        except:
            import traceback
            traceback.print_exc()
        Index += 1

    if ButtonBackPressed == True and Index > 0:
        Index -= 1

    Frame[0:Background.shape[0], 0:round(Background.shape[1] * 0.7)] = Image

    LastLeftclicked = Leftclicked
    LastRightClicked = RightClicked
    AllowAddingPoints = True

    cv2.imshow("Road Segmentation - Annotation", Frame)
    cv2.waitKey(1)

    TimeToSleep = 1/FPS - (time.time() - Start)
    if TimeToSleep > 0:
        time.sleep(TimeToSleep)