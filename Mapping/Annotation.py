from ctypes import windll, byref, sizeof, c_int
import win32con
import win32gui
import shutil
import ctypes
import mouse
import numpy
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
ForegroundWindow = False
LeftClicked = False
RightClicked = False
LastLeftClicked = False
LastRightClicked = False
LastWindowSize = None, None
AllowAddingPoints = True
GrabbedPoint = None, None
Masks = [[(0.45, 0.55), (0.55, 0.55), (0.5, 0.45)]]
RemoveList = []


print("Caching Images...")

Images = []
for File in os.listdir(DATA_PATH):
    if (File.endswith(".png") or File.endswith(".jpg") or File.endswith(".jpeg")) and os.path.exists(f"{DST_PATH}{File.replace('.png', '#SOURCE.png').replace('.jpg', '#SOURCE.jpg').replace('.jpeg', '#SOURCE.jpeg')}") == False:
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


def GetTextSize(Text="NONE", TextWidth=100, Fontsize=11):
    Fontscale = 1
    Textsize, _ = cv2.getTextSize(Text, cv2.FONT_HERSHEY_SIMPLEX, Fontscale, 1)
    WidthCurrentText, HeightCurrentText = Textsize
    maxCountCurrentText = 3
    while WidthCurrentText != TextWidth or HeightCurrentText > Fontsize:
        Fontscale *= min(TextWidth / Textsize[0], Fontsize / Textsize[1])
        Textsize, _ = cv2.getTextSize(Text, cv2.FONT_HERSHEY_SIMPLEX, Fontscale, 1)
        maxCountCurrentText -= 1
        if maxCountCurrentText <= 0:
            break
    Thickness = round(Fontscale * 2)
    if Thickness <= 0:
        Thickness = 1
    return Text, Fontscale, Thickness, Textsize[0], Textsize[1]


def Button(Text="NONE", X1=0, Y1=0, X2=100, Y2=100, Fontsize=11, RoundCorners=5, ButtonSelected=False, TextColor=(255, 255, 255), ButtonColor=(42, 42, 42), ButtonHoverColor=(47, 47, 47), ButtonSelectedColor=(28, 28, 28), ButtonSelectedHoverColor=(28, 28, 28)):
    if X1 <= MouseX * FrameWidth <= X2 and Y1 <= MouseY * FrameHeight <= Y2 and ForegroundWindow:
        ButtonHovered = True
    else:
        ButtonHovered = False
    if ButtonSelected == True:
        if ButtonHovered == True:
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonSelectedHoverColor, RoundCorners, cv2.LINE_AA)
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonSelectedHoverColor, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonSelectedColor, RoundCorners, cv2.LINE_AA)
            cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonSelectedColor, -1, cv2.LINE_AA)
    elif ButtonHovered == True:
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonHoverColor, RoundCorners, cv2.LINE_AA)
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonHoverColor, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonColor, RoundCorners, cv2.LINE_AA)
        cv2.rectangle(Frame, (round(X1+RoundCorners/2), round(Y1+RoundCorners/2)), (round(X2-RoundCorners/2), round(Y2-RoundCorners/2)), ButtonColor, -1, cv2.LINE_AA)
    Text, Fontscale, Thickness, Width, Height = GetTextSize(Text, round((X2-X1)), Fontsize)
    cv2.putText(Frame, Text, (round(X1 + (X2-X1) / 2 - Width / 2), round(Y1 + (Y2-Y1) / 2 + Height / 2)), cv2.FONT_HERSHEY_SIMPLEX, Fontscale, TextColor, Thickness, cv2.LINE_AA)
    if X1 <= MouseX * FrameWidth <= X2 and Y1 <= MouseY * FrameHeight <= Y2 and LeftClicked == False and LastLeftClicked == True:
        return True, LeftClicked and ButtonHovered, ButtonHovered
    else:
        return False, LeftClicked and ButtonHovered, ButtonHovered


def Label(Text="NONE", X1=0, Y1=0, X2=100, Y2=100, Align="Center", Fontsize=11, TextColor=(255, 255, 255)):
    Texts = Text.split("\n")
    LineHeight = ((Y2-Y1) / len(Texts))
    for i, t in enumerate(Texts):
        Text, Fontscale, Thickness, Width, Height = GetTextSize(t, round((X2-X1)), LineHeight / 1.5 if LineHeight / 1.5 < Fontsize else Fontsize)
        if Align == "Center":
            x = round(X1 + (X2-X1) / 2 - Width / 2)
        elif Align == "Left":
            x = round(X1)
        elif Align == "Right":
            x = round(X1 + (X2-X1) - Width)
        cv2.putText(Frame, Text, (x, round(Y1 + (i + 0.5) * LineHeight + Height / 2)), cv2.FONT_HERSHEY_SIMPLEX, Fontscale, TextColor, Thickness, cv2.LINE_AA)


def Save(Index):
    global Masks
    try:
        shutil.copy2(f"{DATA_PATH}{Images[Index][1]}", f"{DST_PATH}{Images[Index][1].replace('.png', '#SOURCE.png').replace('.jpg', '#SOURCE.jpg').replace('.jpeg', '#SOURCE.jpeg')}")
        SouceImageShape = cv2.imread(f"{DATA_PATH}{Images[Index][1]}").shape
        ImageMask = numpy.zeros((SouceImageShape[0], SouceImageShape[1]), numpy.uint8)
        for Mask in Masks:
            ScaledMask = [(round(Point[0] * SouceImageShape[1]), round(Point[1] * SouceImageShape[0])) for Point in Mask]
            cv2.fillPoly(ImageMask, numpy.int32([ScaledMask]), (255, 255, 255), cv2.LINE_AA)
        cv2.imwrite(f"{DST_PATH}{Images[Index][1].replace('.png', '#LABEL.png').replace('.jpg', '#LABEL.jpg').replace('.jpeg', '#LABEL.jpeg')}", ImageMask)
        with open(f"{DST_PATH}{Images[Index][1].replace('.png', '#ANNOTATION.txt').replace('.jpg', '#ANNOTATION.txt').replace('.jpeg', '#ANNOTATION.txt')}", "w") as F:
            F.write(str(Masks))
        Masks = [[(0.45, 0.55), (0.55, 0.55), (0.5, 0.45)]]
    except:
        import traceback
        traceback.print_exc()


def Load(Index):
    global Masks
    try:
        if os.path.exists(f"{DST_PATH}{Images[Index][1].replace('.png', '#ANNOTATION.txt').replace('.jpg', '#ANNOTATION.txt').replace('.jpeg', '#ANNOTATION.txt')}"):
            Masks = eval(open(f"{DST_PATH}{Images[Index][1].replace('.png', '#ANNOTATION.txt').replace('.jpg', '#ANNOTATION.txt').replace('.jpeg', '#ANNOTATION.txt')}", "r").read())
    except:
        import traceback
        traceback.print_exc()


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
            Background = numpy.zeros((WindowHeight, WindowWidth, 3), numpy.uint8)
        MouseX, MouseY = mouse.get_position()
        MouseRelativeWindow = MouseX - WindowX, MouseY - WindowY
        if WindowWidth != 0 and WindowHeight != 0:
            MouseX = MouseRelativeWindow[0] / WindowWidth
            MouseY = MouseRelativeWindow[1] / WindowHeight
        else:
            MouseX = 0
            MouseY = 0
        MouseXImage = MouseX / 0.8
        MouseYImage = MouseY
        ForegroundWindow = ctypes.windll.user32.GetForegroundWindow() == ctypes.windll.user32.FindWindowW(None, "Road Segmentation - Annotation")
    except:
        exit()

    if ctypes.windll.user32.GetKeyState(0x01) & 0x8000 != 0 and ForegroundWindow:
        LeftClicked = True
    else:
        LeftClicked = False

    if ctypes.windll.user32.GetKeyState(0x02) & 0x8000 != 0 and ForegroundWindow:
        RightClicked = True
    else:
        RightClicked = False

    Frame = Background.copy()
    FrameHeight, FrameWidth, _ = Frame.shape
    Image = cv2.resize(Images[Index][0], (round(Background.shape[1] * 0.8), Background.shape[0]))

    Top = 0
    Left = 0
    Bottom = FrameHeight - 1
    Right = FrameWidth - 1

    ButtonNextPressed, _, _ = Button(Text="Next",
                                     X1=0.90 * Right + 2,
                                     Y1=4,
                                     X2=Right - 4,
                                     Y2=0.1 * Bottom - 2)

    ButtonBackPressed, _, _ = Button(Text="Back",
                                     X1=0.80 * Right + 4,
                                     Y1=4,
                                     X2=0.90 * Right - 2,
                                     Y2=0.1 * Bottom - 2)

    ButtonAddPressed, _, _ = Button(Text="Add New Mask",
                                    X1=0.80 * Right + 4,
                                    Y1=0.1 * Bottom + 2,
                                    X2=Right - 4,
                                    Y2=0.2 * Bottom - 4)

    Label(f"Status: {Index + 1} of {len(Images)}", X1=0.80 * Right, Y1=0.2 * Bottom, X2=Right, Y2=0.3 * Bottom)


    if ButtonNextPressed and Index < len(Images) - 1:
        Save(Index)
        Index += 1
        Load(Index)

    if ButtonBackPressed and Index > 0:
        Save(Index)
        Index -= 1
        Load(Index)

    if ButtonAddPressed:
        Masks.append([(0.4, 0.6), (0.6, 0.6), (0.5, 0.5)])


    for i, Mask in enumerate(Masks):
        ScaledMask = [(round(Point[0] * Image.shape[1]), round(Point[1] * Image.shape[0])) for Point in Mask]
        TempImage = Image.copy()
        cv2.fillPoly(TempImage, numpy.int32([ScaledMask]), (0, 200, 100), cv2.LINE_AA)
        Image = cv2.addWeighted(TempImage, 0.2, Image, 0.8, 0)
        cv2.polylines(Image, numpy.int32([ScaledMask]), True, (0, 200, 100), round(WindowHeight/500) if round(WindowHeight/500) > 1 else 1, cv2.LINE_AA)

        for j, Point in enumerate(Mask):
            X, Y = Point
            Radius = round(WindowHeight / 100)
            Radius = 1 if Radius < 1 else Radius
            if GrabbedPoint != (None, None):
                PointGrabbed = True if (i, j) == GrabbedPoint and LeftClicked else False
            else:
                PointGrabbed = True if X * Image.shape[1] - Radius <= MouseXImage * Image.shape[1] <= X * Image.shape[1] + Radius and Y * Image.shape[0] - Radius <= MouseYImage * Image.shape[0] <= Y * Image.shape[0] + Radius and LeftClicked else False
            if LeftClicked == False:
                PointGrabbed = False
                GrabbedPoint = None, None
            if PointGrabbed:
                AllowAddingPoints = False
                GrabbedPoint = i, j
                X = min(1, max(0, MouseXImage))
                Y = min(1, max(0, MouseYImage))
                Masks[i][j] = (X, Y)
                cv2.circle(Image, (round(X * Image.shape[1]), round(Y * Image.shape[0])), Radius, (220, 220, 220), -1, cv2.LINE_AA)
            elif X * Image.shape[1] - Radius <= MouseXImage * Image.shape[1] <= X * Image.shape[1] + Radius and Y * Image.shape[0] - Radius <= MouseYImage * Image.shape[0] <= Y * Image.shape[0] + Radius:
                cv2.circle(Image, (round(X * Image.shape[1]), round(Y * Image.shape[0])), Radius, (20, 220, 120), -1, cv2.LINE_AA)
                if RightClicked == True and LastRightClicked == False:
                    RemoveList.append((i, j))
                AllowAddingPoints = False
            else:
                cv2.circle(Image, (round(X * Image.shape[1]), round(Y * Image.shape[0])), round(Radius * 0.7) if round(Radius * 0.7) > 1 else 1, (0, 200, 100), -1, cv2.LINE_AA)

        for j in range(len(Mask)):
            X1, Y1 = Mask[j]
            if j == 0:
                X2, Y2 = Mask[-1]
            else:
                X2, Y2 = Mask[j - 1]
            VectorX = X2 - X1
            VectorY = Y2 - Y1
            Lenght = math.sqrt(VectorX ** 2 + VectorY ** 2)
            if Lenght == 0:
                Lenght = 0.0001
            Projected = ((MouseXImage - X1) * VectorX + (MouseYImage - Y1) * VectorY) / Lenght
            Projected = max(0, min(1, Projected / Lenght))
            ClosestX = X1 + Projected * VectorX
            ClosestY = Y1 + Projected * VectorY
            Distance = math.sqrt((MouseXImage - ClosestX) ** 2 + (MouseYImage - ClosestY) ** 2)
            if Distance < 0.005:
                cv2.line(Image, (round(X1 * Image.shape[1]), round(Y1 * Image.shape[0])), (round(X2 * Image.shape[1]), round(Y2 * Image.shape[0])), (0, 200, 100), round(WindowHeight/300) if round(WindowHeight/300) > 1 else 1, cv2.LINE_AA)
                if LeftClicked == True and LastLeftClicked == False and AllowAddingPoints == True:
                    AllowAddingPoints = False
                    Masks[i].insert(j, (ClosestX, ClosestY))

    if len(RemoveList) > 0:
        for i, j in sorted(RemoveList, reverse=True):
            if i < len(Masks[i]) and j < len(Masks[i]):
                Masks[i].pop(j)
                if len(Masks[i]) <= 2:
                    Masks.pop(i)
        RemoveList.clear()

    AllowAddingPoints = True


    Frame[0:Background.shape[0], 0:round(Background.shape[1] * 0.8)] = Image

    LastLeftClicked = LeftClicked
    LastRightClicked = RightClicked

    cv2.imshow("Road Segmentation - Annotation", Frame)
    cv2.waitKey(1)

    TimeToSleep = 1 / FPS - (time.time() - Start)
    if TimeToSleep > 0:
        time.sleep(TimeToSleep)