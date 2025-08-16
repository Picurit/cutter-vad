/**
 * Audio source that captures from microphone in real-time.
 */

import javax.sound.sampled.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import java.util.logging.Level;

public class RealTimeMicSource extends AudioSource {
    private static final Logger logger = Logger.getLogger(RealTimeMicSource.class.getName());
    
    private TargetDataLine microphone;
    private AtomicBoolean isCapturing;
    private BlockingQueue<byte[]> frameQueue;
    private Thread captureThread;
    
    public RealTimeMicSource(int sampleRate, int channels, int frameDurationMs) throws Exception {
        super(sampleRate, channels, frameDurationMs);
        
        this.isCapturing = new AtomicBoolean(false);
        this.frameQueue = new LinkedBlockingQueue<>();
        
        initializeMicrophone();
    }
    
    private void initializeMicrophone() throws Exception {
        // Create audio format
        AudioFormat format = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                sampleRate,
                16, // 16-bit
                channels,
                channels * 2, // frame size in bytes
                sampleRate, // frame rate
                false // little-endian
        );
        
        // Get microphone
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
        
        if (!AudioSystem.isLineSupported(info)) {
            throw new UnsupportedAudioFileException("Microphone format not supported");
        }
        
        microphone = (TargetDataLine) AudioSystem.getLine(info);
        microphone.open(format, frameSizeBytes * 4); // Buffer for 4 frames
        
        logger.info("Microphone initialized: " + format);
    }
    
    private void startCapture() {
        if (isCapturing.compareAndSet(false, true)) {
            microphone.start();
            
            captureThread = new Thread(() -> {
                byte[] buffer = new byte[frameSizeBytes];
                
                while (isCapturing.get()) {
                    try {
                        int bytesRead = microphone.read(buffer, 0, frameSizeBytes);
                        
                        if (bytesRead == frameSizeBytes) {
                            // Copy buffer to avoid reference issues
                            byte[] frame = new byte[frameSizeBytes];
                            System.arraycopy(buffer, 0, frame, 0, frameSizeBytes);
                            
                            // Add to queue (non-blocking)
                            if (!frameQueue.offer(frame)) {
                                logger.warning("Frame queue full, dropping frame");
                            }
                        }
                    } catch (Exception e) {
                        if (isCapturing.get()) {
                            logger.log(Level.SEVERE, "Error during microphone capture", e);
                        }
                        break;
                    }
                }
            });
            
            captureThread.setDaemon(true);
            captureThread.start();
            
            logger.info("Microphone capture started");
        }
    }
    
    private void stopCapture() {
        if (isCapturing.compareAndSet(true, false)) {
            if (microphone != null) {
                microphone.stop();
                microphone.close();
            }
            
            if (captureThread != null) {
                try {
                    captureThread.join(1000); // Wait up to 1 second
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            
            logger.info("Microphone capture stopped");
        }
    }
    
    @Override
    public Iterable<byte[]> generateFrames() throws Exception {
        startCapture();
        
        return new Iterable<byte[]>() {
            @Override
            public java.util.Iterator<byte[]> iterator() {
                return new java.util.Iterator<byte[]>() {
                    @Override
                    public boolean hasNext() {
                        return isCapturing.get();
                    }
                    
                    @Override
                    public byte[] next() {
                        try {
                            byte[] frame = frameQueue.take(); // Blocking wait
                            return frame;
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            stopCapture();
                            throw new RuntimeException("Interrupted during frame generation", e);
                        }
                    }
                };
            }
        };
    }
    
    public void cleanup() {
        stopCapture();
    }
}
