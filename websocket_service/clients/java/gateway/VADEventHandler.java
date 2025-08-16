/**
 * VAD Event Handler for Java WebSocket VAD Client
 * 
 * Handles VAD events received from the server and provides statistics.
 */

import java.util.*;
import java.text.SimpleDateFormat;
import java.util.logging.Logger;

public class VADEventHandler {
    private static final Logger logger = Logger.getLogger(VADEventHandler.class.getName());
    
    private List<Map<String, Object>> events;
    private int segmentCount;
    private SimpleDateFormat timeFormatter;
    
    public VADEventHandler() {
        this.events = new ArrayList<>();
        this.segmentCount = 0;
        this.timeFormatter = new SimpleDateFormat("HH:mm:ss");
    }
    
    /**
     * Handle a VAD event from the server.
     */
    public void handleEvent(Map<String, Object> eventData) {
        String eventType = (String) eventData.get("event");
        Object timestampObj = eventData.get("timestamp_ms");
        long timestamp = 0;
        
        if (timestampObj instanceof Number) {
            timestamp = ((Number) timestampObj).longValue();
        }
        
        // Store event
        events.add(new HashMap<>(eventData));
        
        // Log event
        switch (eventType) {
            case "VOICE_START":
                Object segmentIndexObj = eventData.get("segment_index");
                int segmentIndex = segmentIndexObj instanceof Number ? ((Number) segmentIndexObj).intValue() : 0;
                logger.info(String.format("🎙️  VOICE START - Segment #%d at %s", 
                    segmentIndex, formatTimestamp(timestamp)));
                break;
                
            case "VOICE_END":
                segmentIndexObj = eventData.get("segment_index");
                segmentIndex = segmentIndexObj instanceof Number ? ((Number) segmentIndexObj).intValue() : 0;
                Object durationObj = eventData.get("duration_ms");
                long duration = durationObj instanceof Number ? ((Number) durationObj).longValue() : 0;
                segmentCount = Math.max(segmentCount, segmentIndex + 1);
                logger.info(String.format("🔴 VOICE END - Segment #%d, Duration: %dms", 
                    segmentIndex, duration));
                break;
                
            case "VOICE_CONTINUE":
                segmentIndexObj = eventData.get("segment_index");
                segmentIndex = segmentIndexObj instanceof Number ? ((Number) segmentIndexObj).intValue() : 0;
                logger.fine(String.format("🔄 VOICE CONTINUE - Segment #%d", segmentIndex));
                break;
                
            case "TIMEOUT":
                String message = (String) eventData.getOrDefault("message", "");
                logger.warning(String.format("⏰ TIMEOUT - %s", message));
                break;
                
            case "ERROR":
                message = (String) eventData.getOrDefault("message", "");
                logger.severe(String.format("❌ ERROR - %s", message));
                break;
                
            case "INFO":
                message = (String) eventData.getOrDefault("message", "");
                logger.info(String.format("ℹ️  INFO - %s", message));
                break;
                
            default:
                logger.fine("Unknown event type: " + eventType);
                break;
        }
    }
    
    private String formatTimestamp(long timestampMs) {
        return timeFormatter.format(new Date(timestampMs));
    }
    
    /**
     * Get list of detected voice segments.
     */
    public List<Map<String, Object>> getVoiceSegments() {
        List<Map<String, Object>> segments = new ArrayList<>();
        
        // Group START/END events by segment_index
        Map<Integer, Map<String, Object>> startEvents = new HashMap<>();
        Map<Integer, Map<String, Object>> endEvents = new HashMap<>();
        
        for (Map<String, Object> event : events) {
            String eventType = (String) event.get("event");
            Object segmentIndexObj = event.get("segment_index");
            
            if (segmentIndexObj instanceof Number) {
                int segmentIndex = ((Number) segmentIndexObj).intValue();
                
                if ("VOICE_START".equals(eventType)) {
                    startEvents.put(segmentIndex, event);
                } else if ("VOICE_END".equals(eventType)) {
                    endEvents.put(segmentIndex, event);
                }
            }
        }
        
        // Combine start and end events
        Set<Integer> allSegmentIndices = new HashSet<>();
        allSegmentIndices.addAll(startEvents.keySet());
        allSegmentIndices.addAll(endEvents.keySet());
        
        List<Integer> sortedIndices = new ArrayList<>(allSegmentIndices);
        Collections.sort(sortedIndices);
        
        for (int segmentIndex : sortedIndices) {
            Map<String, Object> startEvent = startEvents.get(segmentIndex);
            Map<String, Object> endEvent = endEvents.get(segmentIndex);
            
            if (startEvent != null && endEvent != null) {
                Map<String, Object> segment = new HashMap<>();
                segment.put("segment_index", segmentIndex);
                segment.put("start_ms", startEvent.get("timestamp_ms"));
                segment.put("end_ms", endEvent.get("timestamp_ms"));
                segment.put("duration_ms", endEvent.getOrDefault("duration_ms", 0));
                segments.add(segment);
            }
        }
        
        return segments;
    }
    
    /**
     * Get processing statistics.
     */
    public Map<String, Object> getStatistics() {
        List<Map<String, Object>> segments = getVoiceSegments();
        
        Map<String, Object> statistics = new HashMap<>();
        statistics.put("total_events", events.size());
        statistics.put("voice_segments", segments.size());
        
        long totalVoiceDuration = 0;
        for (Map<String, Object> segment : segments) {
            Object durationObj = segment.get("duration_ms");
            if (durationObj instanceof Number) {
                totalVoiceDuration += ((Number) durationObj).longValue();
            }
        }
        statistics.put("total_voice_duration_ms", totalVoiceDuration);
        
        // Count events by type
        Map<String, Integer> eventsByType = new HashMap<>();
        String[] eventTypes = {"VOICE_START", "VOICE_END", "VOICE_CONTINUE", "TIMEOUT", "ERROR", "INFO"};
        
        for (String eventType : eventTypes) {
            int count = 0;
            for (Map<String, Object> event : events) {
                if (eventType.equals(event.get("event"))) {
                    count++;
                }
            }
            eventsByType.put(eventType, count);
        }
        
        statistics.put("events_by_type", eventsByType);
        
        return statistics;
    }
    
    public int getSegmentCount() {
        return segmentCount;
    }
    
    public List<Map<String, Object>> getEvents() {
        return new ArrayList<>(events);
    }
}
