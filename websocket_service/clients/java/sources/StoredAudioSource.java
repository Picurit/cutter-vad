/**
 * Audio source that reads from a stored audio file (WAV format).
 */

import javax.sound.sampled.*;
import java.io.*;
import java.util.logging.Logger;

public class StoredAudioSource extends AudioSource {
    private static final Logger logger = Logger.getLogger(StoredAudioSource.class.getName());
    
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
