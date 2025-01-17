#include "screencapture.h"

int ScreenWidth = GetSystemMetrics(SM_CXSCREEN);
int ScreenHeight = GetSystemMetrics(SM_CYSCREEN);
int CaptureX = 0;
int CaptureY = 0;
int CaptureWidth = ScreenWidth;
int CaptureHeight = ScreenHeight;
HDC hScreenDC = GetDC(NULL);
HDC hMemoryDC = CreateCompatibleDC(hScreenDC);
HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, ScreenWidth, ScreenHeight);

std::unordered_map<int, std::tuple<int, int, int, int>> ScreenDimensionsCache = {};
std::unordered_map<std::string, int> ScreenIndexCache = {};
std::unordered_map<std::string, std::tuple<int, int, int, int>> ValidateCaptureAreaCache = {};
std::unordered_map<std::string, std::tuple<bool, double>> ForegroundWindowCache = {};
std::unordered_map<std::string, std::tuple<int, int, int, int, double>> WindowPositionCache = {};
std::unordered_map<std::string, double> TrackWindowCache = {};


// MARK: GetLatestFrame()
cv::Mat ScreenCapture::GetLatestFrame() {
    SelectObject(hMemoryDC, hBitmap);

    BitBlt(hMemoryDC, 0, 0, CaptureWidth, CaptureHeight, hScreenDC, CaptureX, CaptureY, SRCCOPY);

    cv::Mat Frame(CaptureHeight, CaptureWidth, CV_8UC4);

    BITMAPINFOHEADER bi = {0};
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = CaptureWidth;
    bi.biHeight = -CaptureHeight;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    GetDIBits(hMemoryDC, hBitmap, 0, CaptureHeight, Frame.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    return Frame;
}


// MARK: SetCaptureArea()
void ScreenCapture::SetCaptureArea(int X1, int Y1, int X2, int Y2) {
    if (CaptureX != X1 || CaptureY != Y1 || CaptureWidth != X2 - X1 || CaptureHeight != Y2 - Y1) {
        CaptureX = X1;
        CaptureY = Y1;
        CaptureWidth = X2 - X1;
        CaptureHeight = Y2 - Y1;
        hScreenDC = GetDC(NULL);
        hMemoryDC = CreateCompatibleDC(hScreenDC);
        hBitmap = CreateCompatibleBitmap(hScreenDC, CaptureWidth, CaptureHeight);
    }
}


// MARK: TakeScreenshot()
cv::Mat ScreenCapture::TakeScreenshot(int X1, int Y1, int X2, int Y2) {
    int Display = GetScreenIndex(round((X1 + X2) / 2), round((Y1 + Y2) / 2));
    std::tuple<int, int, int, int> Area = ValidateCaptureArea(Display, X1, Y1, X2, Y2);
    std::tuple<int, int, int, int> ScreenDimensions = GetScreenDimensions(Display);
    X1 = std::get<0>(Area);
    Y1 = std::get<1>(Area);
    X2 = std::get<2>(Area);
    Y2 = std::get<3>(Area);
    HDC hdcScreen = GetDC(NULL);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    int Width = X2 - X1;
    int Height = Y2 - Y1;
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, Width, Height);
    SelectObject(hdcMem, hBitmap);
    BitBlt(hdcMem, 0, 0, Width, Height, hdcScreen, X1, Y1, SRCCOPY);
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = Width;
    bi.biHeight = -Height;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;
    cv::Mat Image(Height, Width, CV_8UC4);
    GetDIBits(hdcMem, hBitmap, 0, Height, Image.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(NULL, hdcScreen);
    return Image;
}


// MARK: GetScreenDimensions()
std::tuple<int, int, int, int> ScreenCapture::GetScreenDimensions(int Display) {
    std::unordered_map<int, std::tuple<int, int, int, int>>::iterator CachedValue = ScreenDimensionsCache.find(Display);
    if (CachedValue != ScreenDimensionsCache.end()) {
        return ScreenDimensionsCache.at(Display);
    }
    int ScreenX = 0;
    int ScreenY = 0;
    int ScreenWidth = 0;
    int ScreenHeight = 0;
    DISPLAY_DEVICE DisplayDevice;
    DisplayDevice.cb = sizeof(DISPLAY_DEVICE);
    if (EnumDisplayDevices(NULL, Display - 1, &DisplayDevice, 0)) {
        DEVMODE DevMode;
        DevMode.dmSize = sizeof(DEVMODE);
        if (EnumDisplaySettings(DisplayDevice.DeviceName, ENUM_CURRENT_SETTINGS, &DevMode)) {
            ScreenWidth = DevMode.dmPelsWidth;
            ScreenHeight = DevMode.dmPelsHeight;
            ScreenX = DevMode.dmPosition.x;
            ScreenY = DevMode.dmPosition.y;
        }
    }
    ScreenDimensionsCache[Display] = std::make_tuple(ScreenX, ScreenY, ScreenWidth, ScreenHeight);
    return {ScreenX, ScreenY, ScreenWidth, ScreenHeight};
}


// MARK: GetScreenIndex()
int ScreenCapture::GetScreenIndex(int X, int Y) {
    std::string Key = std::to_string(X) + "-" + std::to_string(Y);
    std::unordered_map<std::string, int>::iterator CachedValue = ScreenIndexCache.find(Key);
    if (CachedValue != ScreenIndexCache.end()) {
        return ScreenIndexCache.at(Key);
    }
    int ClosestScreenIndex = -1;
    double ClosestDistance = (std::numeric_limits<double>::max)();
    for (int i = 0; i < GetSystemMetrics(SM_CMONITORS); ++i) {
        auto [ScreenX, ScreenY, ScreenWidth, ScreenHeight] = GetScreenDimensions(i + 1);
        int CenterX = (ScreenX + ScreenX + ScreenWidth) / 2;
        int CenterY = (ScreenY + ScreenY + ScreenHeight) / 2;
        double Distance = std::sqrt(std::pow(CenterX - X, 2) + std::pow(CenterY - Y, 2));
        if (Distance < ClosestDistance) {
            ClosestScreenIndex = i + 1;
            ClosestDistance = Distance;
        }
    }
    ScreenIndexCache[Key] = ClosestScreenIndex;
    return ClosestScreenIndex;
}


// MARK: ValidateCaptureArea()
std::tuple<int, int, int, int> ScreenCapture::ValidateCaptureArea(int Display, int X1, int Y1, int X2, int Y2) {
    std::string Key = std::to_string(Display) + "-" + std::to_string(X1) + "-" + std::to_string(Y1) + "-" + std::to_string(X2) + "-" + std::to_string(Y2);
    std::unordered_map<std::string, std::tuple<int, int, int, int>>::iterator CachedValue = ValidateCaptureAreaCache.find(Key);
    if (CachedValue != ValidateCaptureAreaCache.end()) {
        return ValidateCaptureAreaCache.at(Key);
    }
    auto [ScreenX, ScreenY, Width, Height] = GetScreenDimensions(Display);
    X1 = (std::max)(0, (std::min)(ScreenX + Width - 1, X1));
    X2 = (std::max)(0, (std::min)(ScreenX + Width - 1, X2));
    Y1 = (std::max)(0, (std::min)(ScreenY + Height - 1, Y1));
    Y2 = (std::max)(0, (std::min)(ScreenY + Height - 1, Y2));
    if (X1 == X2) {
        if (X1 == 0) {
            X2 = ScreenX + Width - 1;
        } else {
            X1 = 0;
        }
    }
    if (Y1 == Y2) {
        if (Y1 == 0) {
            Y2 = ScreenY + Height - 1;
        } else {
            Y1 = 0;
        }
    }
    ValidateCaptureAreaCache[Key] = std::make_tuple(X1, Y1, X2, Y2);
    return {X1, Y1, X2, Y2};
}


// MARK: IsForegroundWindow()
bool ScreenCapture::IsForegroundWindow(std::string Name, std::vector<std::string> Blacklist) {
    std::string Key = Name;
    for (const std::string& Item : Blacklist) { Key += Item; }
    double CurrentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000000000.0;
    std::unordered_map<std::string, std::tuple<bool, double>>::iterator CachedValue = ForegroundWindowCache.find(Key);
    if (CachedValue != ForegroundWindowCache.end() && CurrentTime < std::get<1>(ForegroundWindowCache.at(Key)) + 0.5) {
        return std::get<0>(ForegroundWindowCache.at(Key));
    }
    HWND ForegroundHWND = GetForegroundWindow();
    if (ForegroundHWND == NULL) {
        return false;
    }
    char WindowText[256];
    GetWindowText(ForegroundHWND, (LPTSTR)WindowText, sizeof(WindowText));
    std::string ForegroundWindowText(WindowText);
    bool ContainsName = Name.empty() || (ForegroundWindowText.find(Name) != std::string::npos);
    bool ContainsBlacklist = std::any_of(Blacklist.begin(), Blacklist.end(), [&ForegroundWindowText](const std::string& BlacklistItem) {
        return !BlacklistItem.empty() && (ForegroundWindowText.find(BlacklistItem) != std::string::npos);
    });
    ForegroundWindowCache[Key] = std::make_tuple(ContainsName && !ContainsBlacklist, CurrentTime);
    return ContainsName && !ContainsBlacklist;
}


// MARK: GetWindowPosition()
std::tuple<int, int, int, int> ScreenCapture::GetWindowPosition(std::string Name, std::vector<std::string> Blacklist) {
    std::string Key = Name;
    for (const std::string& Item : Blacklist) { Key += Item; }
    double CurrentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000000000.0;
    std::unordered_map<std::string, std::tuple<int, int, int, int, double>>::iterator CachedValue = WindowPositionCache.find(Key);
    if (CachedValue != WindowPositionCache.end() && CurrentTime < std::get<4>(WindowPositionCache.at(Key)) + 0.5) {
        return std::make_tuple(std::get<0>(WindowPositionCache.at(Key)), std::get<1>(WindowPositionCache.at(Key)), std::get<2>(WindowPositionCache.at(Key)), std::get<3>(WindowPositionCache.at(Key)));
    }
    HWND hwnd = NULL;
    std::vector<std::pair<HWND, std::string>> TopWindows;
    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        auto& TopWindows = *reinterpret_cast<std::vector<std::pair<HWND, std::string>>*>(lParam);
        char WindowText[256];
        GetWindowText(hwnd, (LPTSTR)WindowText, sizeof(WindowText));
        TopWindows.emplace_back(hwnd, std::string(WindowText));
        return TRUE;
    }, reinterpret_cast<LPARAM>(&TopWindows));
    for (const auto& [hwnd, WindowText] : TopWindows) {
        bool ContainsName = Name.empty() || (WindowText.find(Name) != std::string::npos);
        bool ContainsBlacklist = std::any_of(Blacklist.begin(), Blacklist.end(), [&WindowText](const std::string& BlacklistItem) {
            return !BlacklistItem.empty() && (WindowText.find(BlacklistItem) != std::string::npos);
        });
        if (ContainsName && !ContainsBlacklist) {
            RECT ClientRect;
            GetClientRect(hwnd, &ClientRect);
            POINT TopLeft = {ClientRect.left, ClientRect.top};
            POINT BottomRight = {ClientRect.right, ClientRect.bottom};
            ClientToScreen(hwnd, &TopLeft);
            ClientToScreen(hwnd, &BottomRight);
            int X = TopLeft.x;
            int Y = TopLeft.y;
            int Width = BottomRight.x - TopLeft.x;
            int Height = BottomRight.y - TopLeft.y;
            WindowPositionCache[Key] = std::make_tuple(0, 0, ScreenWidth, ScreenHeight, CurrentTime);
            return {X, Y, X + Width, Y + Height};
        }
    }
    int ScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    int ScreenHeight = GetSystemMetrics(SM_CYSCREEN);
    WindowPositionCache[Key] = std::make_tuple(0, 0, ScreenWidth, ScreenHeight, CurrentTime);
    return {0, 0, ScreenWidth, ScreenHeight};
}


// MARK: TrackWindow()
void ScreenCapture::TrackWindow(std::string Name, std::vector<std::string> Blacklist, int Rate) {
    std::string Key = Name;
    for (const std::string& Item : Blacklist) { Key += Item; }
    double CurrentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000000000.0;
    std::unordered_map<std::string, double>::iterator CachedValue = TrackWindowCache.find(Key);
    if (Rate > 0) {
        if (CachedValue != TrackWindowCache.end() && TrackWindowCache.at(Key) + 1.0 / Rate > CurrentTime) {
            return;
        }
    }

    auto [X1, Y1, X2, Y2] = GetWindowPosition(Name, Blacklist);
    SetCaptureArea(X1, Y1, X2, Y2);

    TrackWindowCache[Key] = CurrentTime;
}