/**
 * Abstract base class for audio sources.
 */

import javax.sound.sampled.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.Logger;

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
