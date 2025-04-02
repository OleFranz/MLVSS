import datetime
print(f"\n----------------------------------------------\n\n\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')}] \033[0mImporting libraries...")

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
Path = os.path.dirname(os.path.dirname(__file__)).replace("\\", "/")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from PIL import Image
import numpy as np
import threading
import random
import shutil
import torch
import copy
import time
import cv2

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epochs = 100
BatchSize = 32
OutputCount = 1
ImageWidth = int(500 / 2)
ImageHeight = int(1400 / 2)
ColorChannels = 1
LearningRate = 0.001
MaxLearningRate = 0.001
Dropout = 0.1
Patience = 10
Shuffle = True
DropLast = True

DataPath = f"{Path}/AdaptiveCruiseControl/Datasets/AnnotatedDataset"
if os.path.exists(DataPath):
    AllFiles = []
    TrainingFiles = []
    ValidationFiles = []
    for File in os.listdir(f"{DataPath}"):
        if File.endswith(".png"):
            AllFiles.append(f"{DataPath}/{File}")
    AllFiles = random.sample(AllFiles, len(AllFiles))
    TrainingFiles = AllFiles[:int(len(AllFiles) * 0.8)]
    ValidationFiles = AllFiles[int(len(AllFiles) * 0.8):]
    ImageCount = len(TrainingFiles) + len(ValidationFiles)
    TrainingDatasetSize = len(TrainingFiles)
    ValidationDatasetSize = len(ValidationFiles)
else:
    print("No dataset found, exiting...")
    exit()

RED = "\033[91m"
GREEN = "\033[92m"
GRAY = "\033[90m"
NORMAL = "\033[0m"
def Timestamp():
    return GRAY + f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + NORMAL

ModelPath = f"{Path}/AdaptiveCruiseControl/Models"
os.makedirs(ModelPath, exist_ok=True)

print("\n----------------------------------------------\n")

print(Timestamp() + f"Using {str(Device).upper()} for training")
print(Timestamp() + "Number of CPU cores:", multiprocessing.cpu_count())
print()
print(Timestamp() + "Training settings:")
print(Timestamp() + "> Epochs:", Epochs)
print(Timestamp() + "> Batch size:", BatchSize)
print(Timestamp() + "> Outputs:", OutputCount)
print(Timestamp() + "> Images:", ImageCount)
print(Timestamp() + "> Image width:", ImageWidth)
print(Timestamp() + "> Image height:", ImageHeight)
print(Timestamp() + "> Color channels:", ColorChannels)
print(Timestamp() + "> Learning rate:", LearningRate)
print(Timestamp() + "> Max learning rate:", MaxLearningRate)
print(Timestamp() + "> Dropout:", Dropout)
print(Timestamp() + "> Patience:", Patience)
print(Timestamp() + "> Shuffle:", Shuffle)
print(Timestamp() + "> Drop last:", DropLast)

class NeuralNetwork(nn.Module):
    def __init__(Self):
        super(NeuralNetwork, Self).__init__()
        
        Self.Conv2d_1 = nn.Conv2d(ColorChannels, 32, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_1 = nn.BatchNorm2d(32)
        Self.ReLU_1 = nn.ReLU()
        Self.MaxPool2d_1 = nn.MaxPool2d((2, 2))

        Self.Conv2d_2 = nn.Conv2d(32, 64, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_2 = nn.BatchNorm2d(64)
        Self.ReLU_2 = nn.ReLU()
        Self.MaxPool2d_2 = nn.MaxPool2d((2, 2))

        Self.Conv2d_3 = nn.Conv2d(64, 128, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_3 = nn.BatchNorm2d(128)
        Self.ReLU_3 = nn.ReLU()
        Self.MaxPool2d_3 = nn.MaxPool2d((2, 2))

        Self.Conv2d_4 = nn.Conv2d(128, 256, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_4 = nn.BatchNorm2d(256)
        Self.ReLU_4 = nn.ReLU()
        Self.MaxPool2d_4 = nn.MaxPool2d((2, 2))

        Self.Conv2d_5 = nn.Conv2d(256, 512, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_5 = nn.BatchNorm2d(512)
        Self.ReLU_5 = nn.ReLU()
        Self.MaxPool2d_5 = nn.MaxPool2d((2, 2))

        Self.Flatten = nn.Flatten()
        Self.Dropout = nn.Dropout(Dropout)
        Self.Linear_1 = nn.Linear(512 * (ImageWidth // 32) * (ImageHeight // 32), 1024, bias=False)
        Self.BatchNorm1d = nn.BatchNorm1d(1024)
        Self.ReLU_4 = nn.ReLU()
        Self.Linear_2 = nn.Linear(1024, OutputCount, bias=False)

    def forward(Self, X):
        X = Self.Conv2d_1(X)
        X = Self.BatchNorm2d_1(X)
        X = Self.ReLU_1(X)
        X = Self.MaxPool2d_1(X)

        X = Self.Conv2d_2(X)
        X = Self.BatchNorm2d_2(X)
        X = Self.ReLU_2(X)
        X = Self.MaxPool2d_2(X)

        X = Self.Conv2d_3(X)
        X = Self.BatchNorm2d_3(X)
        X = Self.ReLU_3(X)
        X = Self.MaxPool2d_3(X)

        X = Self.Conv2d_4(X)
        X = Self.BatchNorm2d_4(X)
        X = Self.ReLU_4(X)
        X = Self.MaxPool2d_4(X)

        X = Self.Conv2d_5(X)
        X = Self.BatchNorm2d_5(X)
        X = Self.ReLU_5(X)
        X = Self.MaxPool2d_5(X)

        X = Self.Flatten(X)
        X = Self.Dropout(X)
        X = Self.Linear_1(X)
        X = Self.BatchNorm1d(X)
        X = Self.ReLU_4(X)
        X = Self.Linear_2(X)
        return X

def main():
    Model = NeuralNetwork().to(Device).to(torch.bfloat16)

    TotalParameters = 0
    for Parameter in Model.parameters():
        TotalParameters += np.prod(Parameter.size())
    TrainableParameters = sum(Parameter.numel() for Parameter in Model.parameters() if Parameter.requires_grad)
    NonTrainableParameters = TotalParameters - TrainableParameters
    BytesPerParameter = next(Model.parameters()).element_size()
    ModelSize = (TotalParameters * BytesPerParameter) / (1024 ** 2)

    print("\n----------------------------------------------\n")

    print(Timestamp() + "Model properties:")
    print(Timestamp() + f"> Total parameters: {TotalParameters}")
    print(Timestamp() + f"> Trainable parameters: {TrainableParameters}")
    print(Timestamp() + f"> Non-trainable parameters: {NonTrainableParameters}")
    print(Timestamp() + f"> Predicted model size: {ModelSize:.2f}MB")

    print("\n----------------------------------------------\n")

    print(Timestamp() + "Loading...")

    if not os.path.exists(f"{Path}/Tensorboard"):
        os.makedirs(f"{Path}/Tensorboard")

    for Obj in os.listdir(f"{Path}/Tensorboard"):
        try:
            shutil.rmtree(f"{Path}/Tensorboard/{Obj}")
        except:
            os.remove(f"{Path}/Tensorboard/{Obj}")

    TensorBoard = SummaryWriter(f"{Path}/Tensorboard", comment="AdaptiveCruiseControlAI Training", flush_secs=20)

    TrainingTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])

    ValidationTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    class CustomDataset(Dataset):
        def __init__(Self, Files:list, Transform:transforms.Compose, Device:torch.device, DType:torch.dtype, Shuffle:bool, BatchSize:int, BatchPreloadCount:int):
            Self.Cache = {}
            Self.LastUsedIndex = None
            Self.UseFiles = []
            Self.Files = Files
            Self.Transform = Transform
            Self.Device = Device
            Self.DType = DType
            Self.Shuffle = Shuffle
            Self.BatchSize = BatchSize
            Self.BatchPreloadCount = BatchPreloadCount

        def __len__(Self):
            return len(Self.Files)

        def CacheIndex(Self, Index):
            if Index == 0:
                Self.Cache = {}
                Self.LastUsedIndex = None
            if str(Index) not in Self.Cache:
                Self.Cache[str(Index)] = {}
                Self.Cache[str(Index)]["FullyCached"] = False
            threading.Thread(target=Self.CacheIndexThread, args=(Index,), daemon=True).start()

        def CacheIndexThread(Self, Index):
            if Index == 0:
                if Self.Shuffle:
                    Self.UseFiles = random.sample(Self.Files, len(Self.Files))
                else:
                    Self.UseFiles = Self.Files
            elif Index >= len(Self.UseFiles):
                return

            File = Self.UseFiles[Index]

            Image = cv2.imread(File, cv2.IMREAD_GRAYSCALE)
            Image = cv2.resize(Image, (ImageWidth, ImageHeight))
            Image = Image / 255.0
            Image = Self.Transform(Image)
            Image = torch.as_tensor(Image, dtype=Self.DType, device=Self.Device)

            with open(File.replace(".png", ".txt"), "r") as File:
                Label = float(File.read().strip())
            Label = torch.as_tensor([Label], dtype=Self.DType, device=Self.Device)

            Self.Cache[str(Index)]["Image"] = Image
            Self.Cache[str(Index)]["Label"] = Label
            Self.Cache[str(Index)]["FullyCached"] = True

        def ClearIndex(Self, Index):
            if str(Index) in Self.Cache:
                del Self.Cache[str(Index)]

        def GetIndex(Self, Index):
            if str(Index) not in Self.Cache:
                Self.CacheIndex(Index)
                for i in range(Self.BatchPreloadCount * Self.BatchSize):
                    Self.CacheIndex(Index + 1 + i)
            Self.CacheIndex(Index + Self.BatchPreloadCount * Self.BatchSize)
            if Self.LastUsedIndex != None:
                Self.ClearIndex(Self.LastUsedIndex)
            while Self.Cache[str(Index)]["FullyCached"] == False:
                time.sleep(0.0001)
            Self.LastUsedIndex = Index
            return Self.Cache[str(Index)]["Image"], Self.Cache[str(Index)]["Label"]

        def __getitem__(Self, Index):
            return Self.GetIndex(Index)

    TrainingDataset = CustomDataset(TrainingFiles, TrainingTransform, Device, torch.bfloat16, Shuffle, BatchSize, 3)
    ValidationDataset = CustomDataset(ValidationFiles, ValidationTransform, Device, torch.bfloat16, Shuffle, BatchSize, 3)

    TrainingDataloader = DataLoader(TrainingDataset, batch_size=BatchSize, shuffle=False, num_workers=0, pin_memory=False, drop_last=DropLast)
    ValidationDataloader = DataLoader(ValidationDataset, batch_size=BatchSize, shuffle=False, num_workers=0, pin_memory=False, drop_last=DropLast)

    Criterion = nn.MSELoss()
    Optimizer = optim.Adam(Model.parameters(), lr=LearningRate)
    Scheduler = lr_scheduler.OneCycleLR(Optimizer, max_lr=MaxLearningRate, steps_per_epoch=len(TrainingDataloader), epochs=Epochs)

    BestValidationLoss = float("inf")
    BestModel = copy.deepcopy(Model).cpu()
    BestModelEpoch = None
    BestModelTrainingLoss = None
    BestModelValidationLoss = None
    Wait = 0

    TrainingTimePrediction = time.perf_counter()
    TrainingStartTime = time.perf_counter()
    EpochTotalTime = 0
    TrainingLoss = 0
    ValidationLoss = 0
    TrainingEpoch = 0
    Step = 0

    print(f"{Timestamp()}Starting training...")
    print("\n-----------------------------------------------------------------------------------------------------------\n")

    global ProgressPrint
    ProgressPrint = "Initializing"
    def TrainingProgressPrint():
        global ProgressPrint
        def NumToStr(Number):
            Number = format(Number, ".15f")
            while len(Number) > 15:
                Number = Number[:-1]
            while len(Number) < 15:
                Number = Number + "0"
            return Number
        while ProgressPrint == "Initializing":
            time.sleep(1)
        LastMessage = ""
        while ProgressPrint == "FirstEpoch":
            Progress = Step / (len(TrainingDataloader) + len(ValidationDataloader))
            if Progress > 1: Progress = 1
            if Progress < 0: Progress = 0
            Progress = "█" * round(Progress * 10) + "░" * (10 - round(Progress * 10))
            Message = f"{Progress} Waiting for the first epoch to finish..."
            print(f"\r{Message}" + (" " * (len(LastMessage) - len(Message)) if len(LastMessage) > len(Message) else ""), end="", flush=True)
            LastMessage = Message
            time.sleep(1)
        while ProgressPrint == "Running":
            Progress = Step / (len(TrainingDataloader) + len(ValidationDataloader))
            if Progress > 1: Progress = 1
            if Progress < 0: Progress = 0
            Progress = "█" * round(Progress * 10) + "░" * (10 - round(Progress * 10))
            EpochTime = round(EpochTotalTime, 2) if EpochTotalTime > 1 else round((EpochTotalTime) * 1000)
            ETA = round((TrainingTimePrediction - TrainingStartTime) / (TrainingEpoch) * Epochs - (TrainingTimePrediction - TrainingStartTime) + (TrainingTimePrediction - time.perf_counter()))
            MINUTE, HOUR, DAY, MONTH, YEAR = 60, 60*60, 24*60*60, 30*24*60*60, 365*24*60*60
            if ETA < MINUTE:
                ETA = f"{ETA:02d}s"
            elif ETA < HOUR:
                ETA = f"{ETA//MINUTE:02d}:{ETA%MINUTE:02d}"
            elif ETA < DAY:
                h, r = divmod(ETA, HOUR)
                ETA = f"{h:02d}:{r//MINUTE:02d}:{r%MINUTE:02d}"
            elif ETA < MONTH:
                d, r = divmod(ETA, DAY)
                h, r = divmod(r, HOUR)
                ETA = f"{d:02d}:{h:02d}:{r//MINUTE:02d}:{r%MINUTE:02d}"
            elif ETA < YEAR:
                mo, r = divmod(ETA, MONTH)
                d, r = divmod(r, DAY)
                h, r = divmod(r, HOUR)
                ETA = f"{mo:02d}:{d:02d}:{h:02d}:{r//MINUTE:02d}:{r%MINUTE:02d}"
            else:
                y, r = divmod(ETA, YEAR)
                mo, r = divmod(r, MONTH)
                d, r = divmod(r, DAY)
                h, r = divmod(r, HOUR)
                ETA = f"{y:02d}:{mo:02d}:{d:02d}:{h:02d}:{r//MINUTE:02d}:{r%MINUTE:02d}"
            Message = f"{Progress} Epoch {TrainingEpoch}, Train Loss: {NumToStr(TrainingLoss)}, Val Loss: {NumToStr(ValidationLoss)}, {EpochTime}{'s' if EpochTotalTime > 1 else 'ms'}/Epoch, ETA: {ETA}"
            print(f"\r{Message}" + (" " * (len(LastMessage) - len(Message)) if len(LastMessage) > len(Message) else ""), end="", flush=True)
            LastMessage = Message
            time.sleep(1)
        if ProgressPrint == "Early Stopped":
            Message = f"Early stopping at Epoch {TrainingEpoch}, Train Loss: {NumToStr(TrainingLoss)}, Val Loss: {NumToStr(ValidationLoss)}"
            print(f"\r{Message}" + (" " * (len(LastMessage) - len(Message)) if len(LastMessage) > len(Message) else ""), end="", flush=True)
        elif ProgressPrint == "Finished":
            Message = f"Finished at Epoch {TrainingEpoch}, Train Loss: {NumToStr(TrainingLoss)}, Val Loss: {NumToStr(ValidationLoss)}"
            print(f"\r{Message}" + (" " * (len(LastMessage) - len(Message)) if len(LastMessage) > len(Message) else ""), end="", flush=True)
        ProgressPrint = "Received"
    threading.Thread(target=TrainingProgressPrint, daemon=True).start()

    ProgressPrint = "FirstEpoch"

    for Epoch, _ in enumerate(range(Epochs), TrainingEpoch + 1):
        EpochTotalStartTime = time.perf_counter()

        EpochTrainingStartTime = time.perf_counter()
        Model.train()
        RunningTrainingLoss = 0.0
        for i, (Images, Labels) in enumerate(TrainingDataloader, 0):
            Optimizer.zero_grad()
            Outputs = Model(Images)
            Loss = Criterion(Outputs, Labels)
            Loss.backward()
            Optimizer.step()
            Scheduler.step()
            RunningTrainingLoss += Loss.item()
            Step += 1
        RunningTrainingLoss /= len(TrainingDataloader)
        EpochTrainingTime = time.perf_counter() - EpochTrainingStartTime

        EpochValidationStartTime = time.perf_counter()
        Model.eval()
        RunningValidationLoss = 0.0
        with torch.no_grad():
            for i, (Images, Labels) in enumerate(ValidationDataloader, 0):
                Outputs = Model(Images)
                Loss = Criterion(Outputs, Labels)
                RunningValidationLoss += Loss.item()
                Step += 1
        RunningValidationLoss /= len(ValidationDataloader)
        EpochValidationTime = time.perf_counter() - EpochValidationStartTime

        TrainingLoss = RunningTrainingLoss
        ValidationLoss = RunningValidationLoss

        TrainingTimePrediction = time.perf_counter()
        TrainingEpoch = Epoch
        Step = 0

        ProgressPrint = "Running"

        if ValidationLoss < BestValidationLoss:
            BestValidationLoss = ValidationLoss
            BestModel = copy.deepcopy(Model).cpu()
            BestModelEpoch = Epoch
            BestModelTrainingLoss = TrainingLoss
            BestModelValidationLoss = ValidationLoss
            Wait = 0
        else:
            Wait += 1

        EpochTotalTime = time.perf_counter() - EpochTotalStartTime

        TensorBoard.add_scalars(f"Stats", {
            "TrainingLoss": TrainingLoss,
            "ValidationLoss": ValidationLoss,
            "EpochTotalTime": EpochTotalTime,
            "EpochTrainingTime": EpochTrainingTime,
            "EpochValidationTime": EpochValidationTime
        }, Epoch)

        if Wait >= Patience and Patience > 0:
            break

    if ProgressPrint != "Early Stopped":
        ProgressPrint = "Finished"
    while ProgressPrint != "Received":
        time.sleep(1)

    print("\n\n-----------------------------------------------------------------------------------------------------------")

    TrainingTime = time.strftime("%H-%M-%S", time.gmtime(time.time() - TrainingStartTime))
    TrainingDate = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print()
    print(Timestamp() + f"Training completed after " + TrainingTime.replace("-", ":"))


    print(Timestamp() + "Saving the last model...")

    torch.cuda.empty_cache()

    MetadataOptimizer = str(Optimizer)
    MetadataCriterion = str(Criterion)
    MetadataModel = str(Model)
    Metadata = (f"Epochs#{Epoch}",
                f"BatchSize#{BatchSize}",
                f"Classes#{OutputCount}",
                f"Outputs#{OutputCount}",
                f"ImageCount#{ImageCount}",
                f"ImageWidth#{ImageWidth}",
                f"ImageHeight#{ImageHeight}",
                f"ColorChannels#{ColorChannels}",
                f"LearningRate#{LearningRate}",
                f"MaxLearningRate#{MaxLearningRate}",
                f"Dropout#{Dropout}",
                f"Patience#{Patience}",
                f"Shuffle#{Shuffle}",
                f"TrainingTime#{TrainingTime}",
                f"TrainingDate#{TrainingDate}",
                f"TrainingDevice#{Device}",
                f"TrainingOS#{os.name}",
                f"Architecture#{MetadataModel}",
                f"TorchVersion#{torch.__version__}",
                f"NumpyVersion#{np.__version__}",
                f"PILVersion#{Image.__version__}",
                f"TrainingTransform#{TrainingTransform}",
                f"ValidationTransform#{ValidationTransform}",
                f"Optimizer#{MetadataOptimizer}",
                f"LossFunction#{MetadataCriterion}",
                f"TrainingDatasetSize#{TrainingDatasetSize}",
                f"ValidationDatasetSize#{ValidationDatasetSize}",
                f"TrainingLoss#{TrainingLoss}",
                f"ValidationLoss#{ValidationLoss}")
    Metadata = {"Metadata": Metadata}
    Metadata = {Data: str(Value).encode("ascii") for Data, Value in Metadata.items()}

    LastModelSaved = False
    for i in range(5):
        try:
            LastModel = torch.jit.script(Model)
            torch.jit.save(LastModel, os.path.join(ModelPath, f"AdaptiveCruiseControlAI-Last-{TrainingDate}.pt"), _extra_files=Metadata)
            LastModelSaved = True
            break
        except:
            print(Timestamp() + "Failed to save the last model. Retrying...")
    print(Timestamp() + "Last model saved successfully.") if LastModelSaved else print(Timestamp() + "Failed to save the last model.")


    print(Timestamp() + "Saving the best model...")

    torch.cuda.empty_cache()

    MetadataOptimizer = str(Optimizer)
    MetadataCriterion = str(Criterion)
    MetadataModel = str(BestModel)
    Metadata = (f"Epochs#{BestModelEpoch}",
                f"BatchSize#{BatchSize}",
                f"Classes#{OutputCount}",
                f"Outputs#{OutputCount}",
                f"ImageCount#{ImageCount}",
                f"ImageWidth#{ImageWidth}",
                f"ImageHeight#{ImageHeight}",
                f"ColorChannels#{ColorChannels}",
                f"LearningRate#{LearningRate}",
                f"MaxLearningRate#{MaxLearningRate}",
                f"Dropout#{Dropout}",
                f"Patience#{Patience}",
                f"Shuffle#{Shuffle}",
                f"TrainingTime#{TrainingTime}",
                f"TrainingDate#{TrainingDate}",
                f"TrainingDevice#{Device}",
                f"TrainingOS#{os.name}",
                f"Architecture#{MetadataModel}",
                f"TorchVersion#{torch.__version__}",
                f"NumpyVersion#{np.__version__}",
                f"PILVersion#{Image.__version__}",
                f"TrainingTransform#{TrainingTransform}",
                f"ValidationTransform#{ValidationTransform}",
                f"Optimizer#{MetadataOptimizer}",
                f"LossFunction#{MetadataCriterion}",
                f"TrainingDatasetSize#{TrainingDatasetSize}",
                f"ValidationDatasetSize#{ValidationDatasetSize}",
                f"TrainingLoss#{BestModelTrainingLoss}",
                f"ValidationLoss#{BestModelValidationLoss}")
    Metadata = {"Metadata": Metadata}
    Metadata = {Data: str(Value).encode("ascii") for Data, Value in Metadata.items()}

    BestModelSaved = False
    for i in range(5):
        try:
            BestModel = torch.jit.script(BestModel)
            torch.jit.save(BestModel, os.path.join(ModelPath, f"AdaptiveCruiseControlAI-Best-{TrainingDate}.pt"), _extra_files=Metadata)
            BestModelSaved = True
            break
        except:
            print(Timestamp() + "Failed to save the best model. Retrying...")
    print(Timestamp() + "Best model saved successfully.") if BestModelSaved else print(Timestamp() + "Failed to save the best model.")

    print("\n----------------------------------------------\n")

if __name__ == "__main__":
    main()