/*
 * Start-UI.exe — thin console wrapper around Start-UI.bat
 * so users can double-click an .exe next to the batch file.
 * Build: x86_64-w64-mingw32-gcc -O2 -municode -o Start-UI.exe launcher.c
 */
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>

int wmain(void)
{
    wchar_t dir[MAX_PATH];
    wchar_t bat[MAX_PATH];
    wchar_t cmd[MAX_PATH * 2];
    DWORD len;
    wchar_t *slash;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    DWORD code = 1;

    len = GetModuleFileNameW(NULL, dir, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        fwprintf(stderr, L"Cannot locate Start-UI.exe\n");
        return 1;
    }
    slash = wcsrchr(dir, L'\\');
    if (slash) {
        *slash = L'\0';
    }
    SetCurrentDirectoryW(dir);

    if (swprintf(bat, MAX_PATH, L"%s\\Start-UI.bat", dir) < 0) {
        return 1;
    }
    if (GetFileAttributesW(bat) == INVALID_FILE_ATTRIBUTES) {
        fwprintf(stderr, L"Start-UI.bat not found next to Start-UI.exe\n");
        system("pause");
        return 1;
    }
    /* cmd.exe /c "<dir>\Start-UI.bat"  — quotes handle spaces in the path */
    if (swprintf(cmd, MAX_PATH * 2, L"cmd.exe /c \"\"%s\"\"", bat) < 0) {
        return 1;
    }

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, 0, NULL, dir, &si, &pi)) {
        fwprintf(stderr, L"Failed to launch Start-UI.bat (error %lu)\n", GetLastError());
        system("pause");
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}
