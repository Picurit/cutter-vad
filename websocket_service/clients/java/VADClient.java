/**
 * Java WebSocket VAD Client
 * 
 * Main client implementation that combines audio sources and WebSocket gateway
 * for real-time Voice Activity Detection via WebSocket server.
 * 
 * Note: This implementation only supports PCM mode due to the constraint
 * of using only standard Java libraries.
 */

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;
import java.util.logging.Level;

public class VADClient {
    private static final Logger logger = Logger.getLogger(VADClient.class.getName());
    
    private String serverUrl;
    private Map<String, String> config;
    
    private VADEventHandler eventHandler;
    private WebSocketGateway gateway;
    
    public VADClient(String serverUrl, Map<String, String> config) {
        this.serverUrl = serverUrl;
        this.config = config != null ? new HashMap<>(config) : getDefaultConfig();
        
        // Create event handler
        this.eventHandler = new VADEventHandler();
        
        // Create gateway
        this.gateway = new WebSocketGateway(serverUrl, this.config, this.eventHandler);
    }
    
    private Map<String, String> getDefaultConfig() {
        Map<String, String> defaultConfig = new HashMap<>();
        defaultConfig.put("mode", "pcm");
        defaultConfig.put("sample_rate", "16000");
        defaultConfig.put("channels", "1");
        defaultConfig.put("sample_width", "2");
        defaultConfig.put("frame_duration_ms", "30");
        defaultConfig.put("start_probability", "0.4");
        defaultConfig.put("end_probability", "0.3");
        defaultConfig.put("start_frame_count", "6");
        defaultConfig.put("end_frame_count", "12");
        defaultConfig.put("timeout", "30.0");
        return defaultConfig;
    }
    
    /**
     * Process audio from a stored file.
     */
    public Map<String, Object> processStoredAudio(String audioFile, 
                                                 Double durationSeconds, 
                                                 boolean loop) throws Exception {
        logger.info("Processing stored audio: " + audioFile);
        
        // Create audio source
        int sampleRate = Integer.parseInt(config.get("sample_rate"));
        int channels = Integer.parseInt(config.get("channels"));
        int frameDurationMs = Integer.parseInt(config.get("frame_duration_ms"));
        
        StoredAudioSource audioSource = new StoredAudioSource(
            audioFile, sampleRate, channels, frameDurationMs, loop
        );
        
        // Process audio
        long startTime = System.currentTimeMillis();
        
        try {
            return processWithSource(audioSource, durationSeconds);
            
        } finally {
            // Processing completed
            long processingTime = System.currentTimeMillis() - startTime;
            logger.info(String.format("Processing completed in %.2f seconds", processingTime / 1000.0));
        }
    }
    
    /**
     * Process audio from microphone.
     */
    public Map<String, Object> processMicrophone(Double durationSeconds) throws Exception {
        logger.info("Processing microphone audio");
        
        // Create audio source
        int sampleRate = Integer.parseInt(config.get("sample_rate"));
        int channels = Integer.parseInt(config.get("channels"));
        int frameDurationMs = Integer.parseInt(config.get("frame_duration_ms"));
        
        RealTimeMicSource audioSource = new RealTimeMicSource(
            sampleRate, channels, frameDurationMs
        );
        
        try {
            return processWithSource(audioSource, durationSeconds);
            
        } finally {
            audioSource.cleanup();
        }
    }
    
    /**
     * Process audio with the given source.
     */
    private Map<String, Object> processWithSource(AudioSource audioSource, 
                                                 Double durationSeconds) throws Exception {
        
        long startTime = System.currentTimeMillis();
        
        // Connect to server
        if (!gateway.connect()) {
            throw new RuntimeException("Failed to connect to WebSocket server");
        }
        
        try {
            // Send initial configuration if needed
            // (Query parameters already sent during connection)
            
            // Stream audio
            int frameCount = 0;
            long maxDurationMs = durationSeconds != null ? (long)(durationSeconds * 1000) : Long.MAX_VALUE;
            
            for (byte[] frameData : audioSource.generateFrames()) {
                // Check duration limit
                long elapsed = System.currentTimeMillis() - startTime;
                if (elapsed >= maxDurationMs) {
                    logger.info(String.format("Reached duration limit: %.1f seconds", durationSeconds));
                    break;
                }
                
                // Send frame
                gateway.sendAudioFrame(frameData);
                frameCount++;
                
                // Log progress occasionally
                if (frameCount % 100 == 0) {
                    logger.fine(String.format("Streamed %d frames in %.1f seconds", 
                        frameCount, elapsed / 1000.0));
                }
                
                // Check if connection is still alive
                if (!gateway.isConnected()) {
                    logger.warning("Connection lost during streaming");
                    break;
                }
            }
            
            // Wait a bit for final events
            Thread.sleep(1000);
            
        } finally {
            gateway.disconnect();
        }
        
        // Get results
        long processingTime = System.currentTimeMillis() - startTime;
        Map<String, Object> statistics = eventHandler.getStatistics();
        List<Map<String, Object>> segments = eventHandler.getVoiceSegments();
        
        Map<String, Object> results = new HashMap<>();
        results.put("processing_time_seconds", processingTime / 1000.0);
        results.put("statistics", statistics);
        results.put("voice_segments", segments);
        results.put("config", new HashMap<>(config));
        
        logger.info(String.format("Processing completed in %.2f seconds", processingTime / 1000.0));
        logger.info("Detected " + segments.size() + " voice segments");
        
        return results;
    }
    
    /**
     * Print processing results in a formatted way.
     */
    public void printResults(Map<String, Object> results) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("VAD PROCESSING RESULTS");
        System.out.println("=".repeat(60));
        
        // Configuration
        System.out.println("Configuration:");
        @SuppressWarnings("unchecked")
        Map<String, Object> config = (Map<String, Object>) results.get("config");
        for (Map.Entry<String, Object> entry : config.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }
        
        // Statistics
        @SuppressWarnings("unchecked")
        Map<String, Object> stats = (Map<String, Object>) results.get("statistics");
        System.out.println("\nStatistics:");
        System.out.println("  Processing Time: " + results.get("processing_time_seconds") + "s");
        System.out.println("  Total Events: " + stats.get("total_events"));
        System.out.println("  Voice Segments: " + stats.get("voice_segments"));
        System.out.println("  Total Voice Duration: " + stats.get("total_voice_duration_ms") + "ms");
        
        System.out.println("  Events by Type:");
        @SuppressWarnings("unchecked")
        Map<String, Integer> eventsByType = (Map<String, Integer>) stats.get("events_by_type");
        for (Map.Entry<String, Integer> entry : eventsByType.entrySet()) {
            if (entry.getValue() > 0) {
                System.out.println("    " + entry.getKey() + ": " + entry.getValue());
            }
        }
        
        // Voice segments
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> segments = (List<Map<String, Object>>) results.get("voice_segments");
        if (!segments.isEmpty()) {
            System.out.println("\nVoice Segments (" + segments.size() + " detected):");
            for (int i = 0; i < segments.size(); i++) {
                Map<String, Object> segment = segments.get(i);
                Object startMsObj = segment.get("start_ms");
                Object endMsObj = segment.get("end_ms");
                Object durationMsObj = segment.get("duration_ms");
                
                double startS = startMsObj instanceof Number ? ((Number)startMsObj).doubleValue() / 1000 : 0;
                double endS = endMsObj instanceof Number ? ((Number)endMsObj).doubleValue() / 1000 : 0;
                double durationS = durationMsObj instanceof Number ? ((Number)durationMsObj).doubleValue() / 1000 : 0;
                
                System.out.printf("  Segment %d: %.2fs - %.2fs (duration: %.2fs)%n", 
                    i + 1, startS, endS, durationS);
            }
        } else {
            System.out.println("\nNo voice segments detected");
        }
        
        System.out.println("=".repeat(60));
    }
    
    /**
     * Validate test results for automated testing.
     */
    public static boolean validateTestResults(Map<String, Object> results, int expectedSegments) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> segments = (List<Map<String, Object>>) results.get("voice_segments");
        int detectedSegments = segments.size();
        
        System.out.println("\nTest Validation:");
        System.out.println("Expected segments: " + expectedSegments);
        System.out.println("Detected segments: " + detectedSegments);
        
        if (detectedSegments == expectedSegments) {
            System.out.println("✅ Test PASSED - Correct number of segments detected");
            return true;
        } else {
            System.out.println("❌ Test FAILED - Expected " + expectedSegments + 
                             " segments, got " + detectedSegments);
            return false;
        }
    }
    
    /**
     * Main method with command-line interface.
     */
    public static void main(String[] args) {
        try {
            // Parse command line arguments
            Map<String, String> options = parseArguments(args);
            
            if (options.containsKey("help")) {
                printUsage();
                System.exit(0);
            }
            
            // Configure logging
            if (options.containsKey("verbose")) {
                Logger.getGlobal().setLevel(Level.FINE);
            }
            
            // Build configuration
            Map<String, String> config = new HashMap<>();
            config.put("mode", "pcm");
            config.put("sample_rate", options.getOrDefault("sample_rate", "16000"));
            config.put("channels", "1");
            config.put("sample_width", "2");
            config.put("frame_duration_ms", "30");
            config.put("start_probability", options.getOrDefault("start_prob", "0.4"));
            config.put("end_probability", options.getOrDefault("end_prob", "0.3"));
            config.put("start_frame_count", options.getOrDefault("start_frames", "6"));
            config.put("end_frame_count", options.getOrDefault("end_frames", "12"));
            config.put("timeout", options.getOrDefault("timeout", "30.0"));
            
            // Create client
            String serverUrl = options.getOrDefault("server", "ws://localhost:8000/vad");
            VADClient client = new VADClient(serverUrl, config);
            
            // Process audio
            Map<String, Object> results;
            String audioFile = options.get("file");
            boolean microphone = options.containsKey("microphone");
            
            if (audioFile != null) {
                // Validate file exists
                if (!new File(audioFile).exists()) {
                    System.err.println("Error: Audio file not found: " + audioFile);
                    System.exit(1);
                }
                
                Double duration = options.containsKey("duration") ? 
                    Double.parseDouble(options.get("duration")) : null;
                boolean loop = options.containsKey("loop");
                
                results = client.processStoredAudio(audioFile, duration, loop);
                
            } else if (microphone) {
                Double duration = options.containsKey("duration") ? 
                    Double.parseDouble(options.get("duration")) : null;
                    
                results = client.processMicrophone(duration);
                
            } else {
                System.err.println("Error: Must specify either --file or --microphone");
                printUsage();
                System.exit(1);
                return;
            }
            
            // Print results
            client.printResults(results);
            
            // Test validation
            if (options.containsKey("test")) {
                if (validateTestResults(results, 4)) {
                    System.exit(0); // Success
                } else {
                    System.exit(1); // Failure
                }
            }
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error: " + e.getMessage(), e);
            System.exit(1);
        }
    }
    
    /**
     * Parse command line arguments.
     */
    private static Map<String, String> parseArguments(String[] args) {
        Map<String, String> options = new HashMap<>();
        
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            
            if (arg.equals("--help") || arg.equals("-h")) {
                options.put("help", "true");
            } else if (arg.equals("--file") || arg.equals("-f")) {
                if (i + 1 < args.length) {
                    options.put("file", args[++i]);
                }
            } else if (arg.equals("--microphone") || arg.equals("-m")) {
                options.put("microphone", "true");
            } else if (arg.equals("--server") || arg.equals("-s")) {
                if (i + 1 < args.length) {
                    options.put("server", args[++i]);
                }
            } else if (arg.equals("--duration") || arg.equals("-d")) {
                if (i + 1 < args.length) {
                    options.put("duration", args[++i]);
                }
            } else if (arg.equals("--loop")) {
                options.put("loop", "true");
            } else if (arg.equals("--sample-rate")) {
                if (i + 1 < args.length) {
                    options.put("sample_rate", args[++i]);
                }
            } else if (arg.equals("--start-prob")) {
                if (i + 1 < args.length) {
                    options.put("start_prob", args[++i]);
                }
            } else if (arg.equals("--end-prob")) {
                if (i + 1 < args.length) {
                    options.put("end_prob", args[++i]);
                }
            } else if (arg.equals("--start-frames")) {
                if (i + 1 < args.length) {
                    options.put("start_frames", args[++i]);
                }
            } else if (arg.equals("--end-frames")) {
                if (i + 1 < args.length) {
                    options.put("end_frames", args[++i]);
                }
            } else if (arg.equals("--timeout")) {
                if (i + 1 < args.length) {
                    options.put("timeout", args[++i]);
                }
            } else if (arg.equals("--test")) {
                options.put("test", "true");
            } else if (arg.equals("--verbose") || arg.equals("-v")) {
                options.put("verbose", "true");
            }
        }
        
        return options;
    }
    
    /**
     * Print usage information.
     */
    private static void printUsage() {
        System.out.println("Java WebSocket VAD Client");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java VADClient [OPTIONS]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --file, -f <path>        Audio file to process");
        System.out.println("  --microphone, -m         Process microphone input");
        System.out.println("  --server, -s <url>       WebSocket server URL (default: ws://localhost:8000/vad)");
        System.out.println("  --duration, -d <sec>     Maximum processing duration in seconds");
        System.out.println("  --loop                   Loop audio file");
        System.out.println("  --sample-rate <rate>     Audio sample rate (default: 16000)");
        System.out.println("  --start-prob <prob>      VAD start probability threshold (default: 0.4)");
        System.out.println("  --end-prob <prob>        VAD end probability threshold (default: 0.3)");
        System.out.println("  --start-frames <count>   VAD start frame count (default: 6)");
        System.out.println("  --end-frames <count>     VAD end frame count (default: 12)");
        System.out.println("  --timeout <sec>          Voice timeout in seconds (default: 30.0)");
        System.out.println("  --test                   Test mode - validate that exactly 4 segments are detected");
        System.out.println("  --verbose, -v            Enable verbose logging");
        System.out.println("  --help, -h               Show this help message");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  # Process stored audio file");
        System.out.println("  java VADClient --file ../../../examples/audios/SampleVoice.wav");
        System.out.println();
        System.out.println("  # Process microphone for 30 seconds");
        System.out.println("  java VADClient --microphone --duration 30");
        System.out.println();
        System.out.println("  # Test mode (validates expected 4 segments)");
        System.out.println("  java VADClient --file ../../../examples/audios/SampleVoice.wav --test");
    }
}
