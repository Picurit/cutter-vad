#!/bin/bash
# Compilation script for Java WebSocket VAD Client
# Run this from the websocket_service directory

echo "Compiling Java WebSocket VAD Client..."

# Create output directory
mkdir -p build

# Compile all Java files
javac -d build -cp . clients/java/sources/*.java clients/java/gateway/*.java clients/java/*.java examples/*.java

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "To run the test client:"
    echo "  java -cp build TestJavaClient"
    echo ""
    echo "To run the main client:"
    echo "  java -cp build VADClient --help"
else
    echo "❌ Compilation failed!"
    echo "Check the error messages above."
    exit 1
fi
