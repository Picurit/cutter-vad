/**
 * WebSocket Gateway for Java VAD Client
 * 
 * Manages WebSocket connections to the VAD server using only standard Java libraries.
 * Note: This implementation uses basic WebSocket support without external libraries.
 */

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import java.util.logging.Level;

public class WebSocketGateway {
    private static final Logger logger = Logger.getLogger(WebSocketGateway.class.getName());
    
    private static final String WEBSOCKET_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    private static final int FRAME_OPCODE_TEXT = 0x1;
    private static final int FRAME_OPCODE_BINARY = 0x2;
    private static final int FRAME_OPCODE_CLOSE = 0x8;
    private static final int FRAME_OPCODE_PING = 0x9;
    private static final int FRAME_OPCODE_PONG = 0xA;
    
    private String serverUrl;
    private Map<String, String> config;
    private VADEventHandler eventHandler;
    
    private Socket socket;
    private InputStream inputStream;
    private OutputStream outputStream;
    private AtomicBoolean connected;
    private AtomicBoolean running;
    private ExecutorService executorService;
    
    public WebSocketGateway(String serverUrl, Map<String, String> config, VADEventHandler eventHandler) {
        this.serverUrl = serverUrl;
        this.config = config != null ? new HashMap<>(config) : new HashMap<>();
        this.eventHandler = eventHandler != null ? eventHandler : new VADEventHandler();
        
        this.connected = new AtomicBoolean(false);
        this.running = new AtomicBoolean(false);
        this.executorService = Executors.newCachedThreadPool();
    }
    
    /**
     * Build WebSocket connection URL with query parameters.
     */
    private String buildConnectionUrl() {
        if (config.isEmpty()) {
            return serverUrl;
        }
        
        StringBuilder queryString = new StringBuilder();
        for (Map.Entry<String, String> entry : config.entrySet()) {
            if (queryString.length() > 0) {
                queryString.append("&");
            }
            try {
                queryString.append(URLEncoder.encode(entry.getKey(), "UTF-8"))
                          .append("=")
                          .append(URLEncoder.encode(entry.getValue(), "UTF-8"));
            } catch (UnsupportedEncodingException e) {
                // UTF-8 is always supported
                throw new RuntimeException(e);
            }
        }
        
        if (queryString.length() > 0) {
            return serverUrl + "?" + queryString.toString();
        } else {
            return serverUrl;
        }
    }
    
    /**
     * Connect to WebSocket server.
     */
    public boolean connect() throws Exception {
        try {
            String connectionUrl = buildConnectionUrl();
            logger.info("Connecting to " + connectionUrl);
            
            // Parse URL
            URI uri = new URI(connectionUrl);
            String host = uri.getHost();
            int port = uri.getPort();
            if (port == -1) {
                port = "wss".equals(uri.getScheme()) ? 443 : 80;
            }
            String path = uri.getPath();
            if (uri.getQuery() != null) {
                path += "?" + uri.getQuery();
            }
            
            // Create socket connection
            socket = new Socket(host, port);
            inputStream = socket.getInputStream();
            outputStream = socket.getOutputStream();
            
            // Perform WebSocket handshake
            if (performHandshake(host, path)) {
                connected.set(true);
                running.set(true);
                
                // Start message listener
                executorService.submit(this::messageListener);
                
                logger.info("WebSocket connection established");
                return true;
            } else {
                disconnect();
                return false;
            }
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Failed to connect", e);
            connected.set(false);
            return false;
        }
    }
    
    /**
     * Perform WebSocket handshake.
     */
    private boolean performHandshake(String host, String path) throws Exception {
        // Generate WebSocket key
        byte[] keyBytes = new byte[16];
        new Random().nextBytes(keyBytes);
        String webSocketKey = Base64.getEncoder().encodeToString(keyBytes);
        
        // Build handshake request
        StringBuilder request = new StringBuilder();
        request.append("GET ").append(path).append(" HTTP/1.1\r\n");
        request.append("Host: ").append(host).append("\r\n");
        request.append("Upgrade: websocket\r\n");
        request.append("Connection: Upgrade\r\n");
        request.append("Sec-WebSocket-Key: ").append(webSocketKey).append("\r\n");
        request.append("Sec-WebSocket-Version: 13\r\n");
        request.append("\r\n");
        
        // Send handshake request
        outputStream.write(request.toString().getBytes(StandardCharsets.UTF_8));
        outputStream.flush();
        
        // Read handshake response
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));
        String statusLine = reader.readLine();
        
        if (statusLine == null || !statusLine.contains("101")) {
            logger.severe("WebSocket handshake failed: " + statusLine);
            return false;
        }
        
        // Read headers
        Map<String, String> headers = new HashMap<>();
        String line;
        while ((line = reader.readLine()) != null && !line.isEmpty()) {
            int colonIndex = line.indexOf(':');
            if (colonIndex > 0) {
                String headerName = line.substring(0, colonIndex).trim().toLowerCase();
                String headerValue = line.substring(colonIndex + 1).trim();
                headers.put(headerName, headerValue);
            }
        }
        
        // Verify handshake response
        String expectedAccept = calculateWebSocketAccept(webSocketKey);
        String actualAccept = headers.get("sec-websocket-accept");
        
        if (!expectedAccept.equals(actualAccept)) {
            logger.severe("WebSocket handshake verification failed");
            return false;
        }
        
        logger.info("WebSocket handshake successful");
        return true;
    }
    
    /**
     * Calculate WebSocket accept value.
     */
    private String calculateWebSocketAccept(String webSocketKey) throws Exception {
        String combined = webSocketKey + WEBSOCKET_MAGIC_STRING;
        MessageDigest digest = MessageDigest.getInstance("SHA-1");
        byte[] hash = digest.digest(combined.getBytes(StandardCharsets.UTF_8));
        return Base64.getEncoder().encodeToString(hash);
    }
    
    /**
     * Disconnect from WebSocket server.
     */
    public void disconnect() {
        running.set(false);
        connected.set(false);
        
        try {
            if (socket != null && !socket.isClosed()) {
                socket.close();
            }
        } catch (IOException e) {
            logger.log(Level.WARNING, "Error closing socket", e);
        }
        
        logger.info("WebSocket connection closed");
    }
    
    /**
     * Send configuration update to server.
     */
    public void sendConfig(Map<String, Object> configUpdate) throws Exception {
        if (!connected.get()) {
            throw new RuntimeException("Not connected to server");
        }
        
        Map<String, Object> configMessage = new HashMap<>();
        configMessage.put("type", "CONFIG");
        configMessage.putAll(configUpdate);
        
        String json = mapToJson(configMessage);
        sendTextFrame(json);
        
        logger.info("Sent configuration update: " + configUpdate);
    }
    
    /**
     * Send audio frame to server.
     */
    public void sendAudioFrame(byte[] frameData) throws Exception {
        if (!connected.get()) {
            throw new RuntimeException("Not connected to server");
        }
        
        sendBinaryFrame(frameData);
    }
    
    /**
     * Send heartbeat message to server.
     */
    public void sendHeartbeat() throws Exception {
        if (!connected.get()) {
            return;
        }
        
        Map<String, Object> heartbeatMessage = new HashMap<>();
        heartbeatMessage.put("type", "HEARTBEAT");
        
        String json = mapToJson(heartbeatMessage);
        sendTextFrame(json);
        
        logger.fine("Sent heartbeat");
    }
    
    /**
     * Send text frame.
     */
    private void sendTextFrame(String text) throws Exception {
        byte[] payload = text.getBytes(StandardCharsets.UTF_8);
        sendFrame(FRAME_OPCODE_TEXT, payload);
    }
    
    /**
     * Send binary frame.
     */
    private void sendBinaryFrame(byte[] data) throws Exception {
        sendFrame(FRAME_OPCODE_BINARY, data);
    }
    
    /**
     * Send WebSocket frame.
     */
    private synchronized void sendFrame(int opcode, byte[] payload) throws Exception {
        if (outputStream == null) {
            throw new IOException("Output stream not available");
        }
        
        // Create frame header
        ByteArrayOutputStream frame = new ByteArrayOutputStream();
        
        // First byte: FIN = 1, RSV = 0, Opcode
        frame.write(0x80 | opcode);
        
        // Payload length and mask
        boolean masked = true; // Client frames should be masked
        int payloadLength = payload.length;
        
        if (payloadLength < 126) {
            frame.write((masked ? 0x80 : 0) | payloadLength);
        } else if (payloadLength < 65536) {
            frame.write((masked ? 0x80 : 0) | 126);
            frame.write((payloadLength >> 8) & 0xFF);
            frame.write(payloadLength & 0xFF);
        } else {
            frame.write((masked ? 0x80 : 0) | 127);
            frame.write(0); // Extended payload length (high 32 bits)
            frame.write(0);
            frame.write(0);
            frame.write(0);
            frame.write((payloadLength >> 24) & 0xFF); // Extended payload length (low 32 bits)
            frame.write((payloadLength >> 16) & 0xFF);
            frame.write((payloadLength >> 8) & 0xFF);
            frame.write(payloadLength & 0xFF);
        }
        
        // Masking key and masked payload
        if (masked) {
            byte[] maskingKey = new byte[4];
            new Random().nextBytes(maskingKey);
            frame.write(maskingKey);
            
            // Mask payload
            for (int i = 0; i < payload.length; i++) {
                frame.write(payload[i] ^ maskingKey[i % 4]);
            }
        } else {
            frame.write(payload);
        }
        
        // Send frame
        outputStream.write(frame.toByteArray());
        outputStream.flush();
    }
    
    /**
     * Message listener thread.
     */
    private void messageListener() {
        try {
            while (running.get() && connected.get()) {
                try {
                    // Read frame
                    WebSocketFrame frame = readFrame();
                    if (frame != null) {
                        handleFrame(frame);
                    }
                } catch (SocketTimeoutException e) {
                    // Timeout is normal, continue
                } catch (IOException e) {
                    if (running.get()) {
                        logger.log(Level.WARNING, "Connection lost", e);
                        connected.set(false);
                    }
                    break;
                }
            }
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error in message listener", e);
            connected.set(false);
        }
    }
    
    /**
     * Read WebSocket frame.
     */
    private WebSocketFrame readFrame() throws IOException {
        if (inputStream.available() < 2) {
            Thread.yield();
            return null;
        }
        
        // Read first two bytes
        int byte1 = inputStream.read();
        int byte2 = inputStream.read();
        
        if (byte1 == -1 || byte2 == -1) {
            throw new IOException("Unexpected end of stream");
        }
        
        boolean fin = (byte1 & 0x80) != 0;
        int opcode = byte1 & 0x0F;
        boolean masked = (byte2 & 0x80) != 0;
        int payloadLength = byte2 & 0x7F;
        
        // Extended payload length
        if (payloadLength == 126) {
            int len1 = inputStream.read();
            int len2 = inputStream.read();
            if (len1 == -1 || len2 == -1) {
                throw new IOException("Unexpected end of stream");
            }
            payloadLength = (len1 << 8) | len2;
        } else if (payloadLength == 127) {
            // Read 8 bytes for extended length (we only support up to 32-bit length)
            for (int i = 0; i < 4; i++) {
                if (inputStream.read() == -1) {
                    throw new IOException("Unexpected end of stream");
                }
            }
            
            int len1 = inputStream.read();
            int len2 = inputStream.read();
            int len3 = inputStream.read();
            int len4 = inputStream.read();
            if (len1 == -1 || len2 == -1 || len3 == -1 || len4 == -1) {
                throw new IOException("Unexpected end of stream");
            }
            payloadLength = (len1 << 24) | (len2 << 16) | (len3 << 8) | len4;
        }
        
        // Masking key
        byte[] maskingKey = null;
        if (masked) {
            maskingKey = new byte[4];
            if (inputStream.read(maskingKey) != 4) {
                throw new IOException("Unexpected end of stream");
            }
        }
        
        // Payload
        byte[] payload = new byte[payloadLength];
        int totalRead = 0;
        while (totalRead < payloadLength) {
            int read = inputStream.read(payload, totalRead, payloadLength - totalRead);
            if (read == -1) {
                throw new IOException("Unexpected end of stream");
            }
            totalRead += read;
        }
        
        // Unmask payload
        if (masked && maskingKey != null) {
            for (int i = 0; i < payload.length; i++) {
                payload[i] ^= maskingKey[i % 4];
            }
        }
        
        return new WebSocketFrame(fin, opcode, payload);
    }
    
    /**
     * Handle received frame.
     */
    private void handleFrame(WebSocketFrame frame) {
        switch (frame.opcode) {
            case FRAME_OPCODE_TEXT:
                String text = new String(frame.payload, StandardCharsets.UTF_8);
                try {
                    Map<String, Object> eventData = jsonToMap(text);
                    eventHandler.handleEvent(eventData);
                } catch (Exception e) {
                    logger.log(Level.WARNING, "Failed to parse JSON message: " + text, e);
                }
                break;
                
            case FRAME_OPCODE_BINARY:
                logger.fine("Received binary frame (unexpected from server)");
                break;
                
            case FRAME_OPCODE_CLOSE:
                logger.info("Received close frame from server");
                connected.set(false);
                break;
                
            case FRAME_OPCODE_PING:
                try {
                    sendFrame(FRAME_OPCODE_PONG, frame.payload);
                } catch (Exception e) {
                    logger.log(Level.WARNING, "Failed to send pong", e);
                }
                break;
                
            case FRAME_OPCODE_PONG:
                logger.fine("Received pong frame");
                break;
                
            default:
                logger.warning("Unknown frame opcode: " + frame.opcode);
                break;
        }
    }
    
    /**
     * Simple JSON to Map conversion (basic implementation).
     */
    private Map<String, Object> jsonToMap(String json) {
        Map<String, Object> result = new HashMap<>();
        
        // Very basic JSON parsing - only supports simple key-value pairs
        json = json.trim();
        if (json.startsWith("{") && json.endsWith("}")) {
            json = json.substring(1, json.length() - 1);
            
            String[] pairs = json.split(",");
            for (String pair : pairs) {
                String[] keyValue = pair.split(":", 2);
                if (keyValue.length == 2) {
                    String key = keyValue[0].trim().replaceAll("\"", "");
                    String value = keyValue[1].trim();
                    
                    // Remove quotes from string values
                    if (value.startsWith("\"") && value.endsWith("\"")) {
                        value = value.substring(1, value.length() - 1);
                        result.put(key, value);
                    } else if ("true".equals(value) || "false".equals(value)) {
                        result.put(key, Boolean.parseBoolean(value));
                    } else {
                        try {
                            if (value.contains(".")) {
                                result.put(key, Double.parseDouble(value));
                            } else {
                                result.put(key, Long.parseLong(value));
                            }
                        } catch (NumberFormatException e) {
                            result.put(key, value);
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Simple Map to JSON conversion (basic implementation).
     */
    private String mapToJson(Map<String, Object> map) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        
        boolean first = true;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (!first) {
                json.append(",");
            }
            first = false;
            
            json.append("\"").append(entry.getKey()).append("\":");
            
            Object value = entry.getValue();
            if (value instanceof String) {
                json.append("\"").append(value).append("\"");
            } else if (value instanceof Number || value instanceof Boolean) {
                json.append(value);
            } else {
                json.append("\"").append(value.toString()).append("\"");
            }
        }
        
        json.append("}");
        return json.toString();
    }
    
    public boolean isConnected() {
        return connected.get();
    }
    
    public VADEventHandler getEventHandler() {
        return eventHandler;
    }
    
    /**
     * Clean up resources.
     */
    public void cleanup() {
        disconnect();
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * WebSocket frame representation.
     */
    private static class WebSocketFrame {
        final boolean fin;
        final int opcode;
        final byte[] payload;
        
        WebSocketFrame(boolean fin, int opcode, byte[] payload) {
            this.fin = fin;
            this.opcode = opcode;
            this.payload = payload;
        }
    }
}
