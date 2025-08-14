/**
 * Audio Sources for Java WebSocket VAD Client
 * 
 * This class provides audio source implementations for capturing or reading
 * audio data and emitting PCM frames for WebSocket transmission.
 * 
 * Note: This implementation only supports PCM mode due to the constraint
 * of using only standard Java libraries (no external audio codecs).
 */

package sources;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Abstract base class for audio sources.
 */
public abstract class AudioSource {
    protected static final Logger logger = Logger.getLogger(AudioSource.class.getName());
    
    protected int sampleRate;
    protected int channels;
    protected int frameDurationMs;
    protected int frameSizeSamples;
    protected int frameSizeBytes;
    
    public AudioSource(int sampleRate, int channels, int frameDurationMs) {
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.frameDurationMs = frameDurationMs;
        
        // Calculate frame size in samples and bytes
        this.frameSizeSamples = (int) (sampleRate * (frameDurationMs / 1000.0));
        this.frameSizeBytes = frameSizeSamples * channels * 2; // 16-bit = 2 bytes per sample
        
        logger.info(String.format("AudioSource initialized: %d Hz, %d channels, %d ms frames (%d samples, %d bytes)",
                sampleRate, channels, frameDurationMs, frameSizeSamples, frameSizeBytes));
    }
    
    /**
     * Generate PCM audio frames as byte arrays.
     * @return Iterator over PCM frame data
     * @throws Exception if frame generation fails
     */
    public abstract Iterable<byte[]> generateFrames() throws Exception;
    
    /**
     * Convert audio samples to PCM bytes (16-bit signed int, little-endian).
     */
    protected byte[] samplesToPcmBytes(float[] samples) {
        ByteBuffer buffer = ByteBuffer.allocate(samples.length * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        for (float sample : samples) {
            // Clamp to [-1, 1] and convert to 16-bit signed int
            float clampedSample = Math.max(-1.0f, Math.min(1.0f, sample));
            short pcmSample = (short) (clampedSample * 32767);
            buffer.putShort(pcmSample);
        }
        
        return buffer.array();
    }
    
    /**
     * Convert 16-bit PCM bytes to float samples.
     */
    protected float[] pcmBytesToSamples(byte[] pcmBytes) {
        ByteBuffer buffer = ByteBuffer.wrap(pcmBytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        float[] samples = new float[pcmBytes.length / 2];
        for (int i = 0; i < samples.length; i++) {
            short pcmSample = buffer.getShort();
            samples[i] = pcmSample / 32768.0f;
        }
        
        return samples;
    }
    
    public int getFrameSizeBytes() {
        return frameSizeBytes;
    }
    
    public int getSampleRate() {
        return sampleRate;
    }
    
    public int getChannels() {
        return channels;
    }
}

/**
 * Audio source that reads from a stored audio file (WAV format).
 */
class StoredAudioSource extends AudioSource {
    private File audioFile;
    private boolean loop;
    private AudioInputStream audioInputStream;
    private float[] audioData;
    private int currentPosition;
    
    public StoredAudioSource(String filePath, int sampleRate, int channels, int frameDurationMs, boolean loop) 
            throws Exception {
        super(sampleRate, channels, frameDurationMs);
        
        this.audioFile = new File(filePath);
        this.loop = loop;
        this.currentPosition = 0;
        
        if (!audioFile.exists()) {
            throw new FileNotFoundException("Audio file not found: " + filePath);
        }
        
        loadAudio();
    }
    
    private void loadAudio() throws Exception {
        logger.info("Loading audio file: " + audioFile.getAbsolutePath());
        
        // Open audio file
        audioInputStream = AudioSystem.getAudioInputStream(audioFile);
        AudioFormat originalFormat = audioInputStream.getFormat();
        
        logger.info(String.format("Original format: %.0f Hz, %d channels, %d bits",
                originalFormat.getSampleRate(), originalFormat.getChannels(), 
                originalFormat.getSampleSizeInBits()));
        
        // Create target format (16-bit PCM, little-endian)
        AudioFormat targetFormat = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                sampleRate,
                16, // 16-bit
                channels,
                channels * 2, // frame size in bytes
                sampleRate, // frame rate
                false // little-endian
        );
        
        // Convert if necessary
        AudioInputStream convertedStream = audioInputStream;
        if (!originalFormat.equals(targetFormat)) {
            if (AudioSystem.isConversionSupported(targetFormat, originalFormat)) {
                convertedStream = AudioSystem.getAudioInputStream(targetFormat, audioInputStream);
                logger.info("Audio format converted to target format");
            } else {
                logger.warning("Audio format conversion not supported - attempting to use original format");
            }
        }
        
        // Read all audio data
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] tempBuffer = new byte[4096];
        int bytesRead;
        
        while ((bytesRead = convertedStream.read(tempBuffer)) != -1) {
            buffer.write(tempBuffer, 0, bytesRead);
        }
        
        byte[] audioBytes = buffer.toByteArray();
        
        // Convert to float samples
        audioData = pcmBytesToSamples(audioBytes);
        
        logger.info(String.format("Loaded %d samples (%.2f seconds)",
                audioData.length, (double) audioData.length / sampleRate));
        
        // Close streams
        convertedStream.close();
        audioInputStream.close();
    }
    
    @Override
    public Iterable<byte[]> generateFrames() throws Exception {
        return new Iterable<byte[]>() {
            @Override
            public java.util.Iterator<byte[]> iterator() {
                return new java.util.Iterator<byte[]>() {
                    @Override
                    public boolean hasNext() {
                        return currentPosition < audioData.length || loop;
                    }
                    
                    @Override
                    public byte[] next() {
                        // Check if we have enough samples for a complete frame
                        int remainingSamples = audioData.length - currentPosition;
                        
                        float[] frameSamples = new float[frameSizeSamples];
                        
                        if (remainingSamples >= frameSizeSamples) {
                            // Extract frame
                            System.arraycopy(audioData, currentPosition, frameSamples, 0, frameSizeSamples);
                            currentPosition += frameSizeSamples;
                            
                        } else if (remainingSamples > 0) {
                            // Pad last frame with zeros
                            System.arraycopy(audioData, currentPosition, frameSamples, 0, remainingSamples);
                            // Rest of frameSamples is already zero-initialized
                            currentPosition = audioData.length;
                            
                        } else {
                            // End of file
                            if (loop) {
                                logger.info("Looping audio file");
                                currentPosition = 0;
                                return next(); // Recursive call to get next frame
                            } else {
                                throw new RuntimeException("No more frames available");
                            }
                        }
                        
                        // Convert to PCM bytes
                        byte[] pcmBytes = samplesToPcmBytes(frameSamples);
                        
                        // Simulate real-time playback
                        try {
                            Thread.sleep(frameDurationMs);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException("Interrupted during frame generation", e);
                        }
                        
                        return pcmBytes;
                    }
                };
            }
        };
    }
}

/**
 * Audio source that captures from microphone in real-time.
 */
class RealTimeMicSource extends AudioSource {
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
    
    @Override
    protected void finalize() throws Throwable {
        stopCapture();
        super.finalize();
    }
}

/**
 * Factory class for creating audio sources.
 */
public class AudioSources {
    /**
     * Create a stored audio source.
     */
    public static StoredAudioSource createStoredAudioSource(String filePath, 
                                                           int sampleRate, 
                                                           int channels, 
                                                           int frameDurationMs, 
                                                           boolean loop) throws Exception {
        return new StoredAudioSource(filePath, sampleRate, channels, frameDurationMs, loop);
    }
    
    /**
     * Create a real-time microphone source.
     */
    public static RealTimeMicSource createRealTimeMicSource(int sampleRate, 
                                                           int channels, 
                                                           int frameDurationMs) throws Exception {
        return new RealTimeMicSource(sampleRate, channels, frameDurationMs);
    }
    
    /**
     * Test function for StoredAudioSource.
     */
    public static void testStoredAudioSource(String audioFilePath) {
        try {
            logger.info("Testing StoredAudioSource with file: " + audioFilePath);
            
            StoredAudioSource source = createStoredAudioSource(audioFilePath, 16000, 1, 30, false);
            
            int frameCount = 0;
            for (byte[] frame : source.generateFrames()) {
                frameCount++;
                System.out.println("Frame " + frameCount + ": " + frame.length + " bytes");
                
                // Stop after 10 frames for testing
                if (frameCount >= 10) {
                    break;
                }
            }
            
            System.out.println("Successfully generated " + frameCount + " frames");
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Test failed", e);
        }
    }
    
    public static void main(String[] args) {
        // Test with sample audio file
        String audioFile = "../../../examples/audios/SampleVoice.wav";
        testStoredAudioSource(audioFile);
    }
}
