@echo off
REM Compilation script for Java WebSocket VAD Client
REM Run this from the websocket_service directory

echo Compiling Java WebSocket VAD Client...

REM Create output directory
if not exist "build" mkdir build

REM Compile all Java files
javac -d build -cp . clients/java/sources/*.java clients/java/gateway/*.java clients/java/*.java examples/*.java

if %ERRORLEVEL% EQU 0 (
    echo ✅ Compilation successful!
    echo.
    echo To run the test client:
    echo   java -cp build TestJavaClient
    echo.
    echo To run the main client:
    echo   java -cp build VADClient --help
) else (
    echo ❌ Compilation failed!
    echo Check the error messages above.
    exit /b 1
)
