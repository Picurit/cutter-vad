"""
Core Silero model wrapper for Voice Activity Detection.
"""

import os
import numpy as np
from typing import Optional, Callable, Any, Dict
from pathlib import Path
import onnxruntime as ort
from collections import deque

from .config import VADConfig, SampleRate, SileroModelVersion
from .exceptions import (
    VADError, 
    ModelNotFoundError, 
    ModelInitializationError,
    AudioProcessingError,
    CallbackError
)
from ..utils.audio import AudioUtils
from ..utils.wav_writer import WAVWriter


class SileroVADModel:
    """Silero VAD model wrapper using ONNX Runtime."""
    
    def __init__(self, model_path: str, model_version: SileroModelVersion) -> None:
        """
        Initialize Silero VAD model.
        
        Args:
            model_path: Path to the ONNX model file
            model_version: Model version (v4 or v5)
        """
        self.model_path = model_path
        self.model_version = model_version
        self.session: Optional[ort.InferenceSession] = None
        self.hidden_state: Optional[np.ndarray] = None
        self.cell_state: Optional[np.ndarray] = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ONNX model."""
        try:
            if not os.path.exists(self.model_path):
                raise ModelNotFoundError(self.model_path)
            
            # Create ONNX Runtime session with optimizations
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers() and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Initialize hidden states
            self._reset_states()
            
        except Exception as e:
            raise ModelInitializationError(
                self.model_version.value,
                f"Failed to load model from {self.model_path}: {str(e)}"
            )
    
    def _reset_states(self) -> None:
        """Reset LSTM hidden states."""
        if self.model_version == SileroModelVersion.V5:
            # V5 uses combined state (2, batch, 128)
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            # Older versions use separate h and c states (2, batch, 64)
            self.hidden_state = np.zeros((2, 1, 64), dtype=np.float32)
            self.cell_state = np.zeros((2, 1, 64), dtype=np.float32)
    
    def predict(self, audio_chunk: np.ndarray, sample_rate: int) -> float:
        """
        Predict voice activity probability for audio chunk.
        
        Args:
            audio_chunk: Audio chunk (512 samples for 16kHz)
            sample_rate: Sample rate of the audio
            
        Returns:
            Voice activity probability (0.0 to 1.0)
        """
        try:
            if self.session is None:
                raise ModelInitializationError(self.model_version.value, "Model not loaded")
            
            # Ensure input is correct shape and type
            if len(audio_chunk) != 512:
                # Pad or truncate to 512 samples
                if len(audio_chunk) < 512:
                    audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:512]
            
            # Reshape for model input: (batch_size, sequence_length)
            input_audio = audio_chunk.reshape(1, -1).astype(np.float32)
            
            # Prepare inputs based on model version
            if self.model_version == SileroModelVersion.V5:
                inputs = {
                    'input': input_audio,
                    'state': self.state,
                    'sr': np.array([sample_rate], dtype=np.int64)
                }
            else:
                inputs = {
                    'input': input_audio,
                    'h': self.hidden_state,
                    'c': self.cell_state,
                    'sr': np.array([sample_rate], dtype=np.int64)
                }
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Extract outputs based on model version
            probability = float(outputs[0][0][0])  # Voice probability
            
            if self.model_version == SileroModelVersion.V5:
                self.state = outputs[1]  # Updated state
            else:
                self.hidden_state = outputs[1]  # Updated hidden state
                self.cell_state = outputs[2]   # Updated cell state
            
            return probability
            
        except Exception as e:
            raise AudioProcessingError(f"Model prediction failed: {str(e)}")
    
    def reset(self) -> None:
        """Reset model states."""
        self._reset_states()


class VADProcessor:
    """Voice Activity Detection processor with state management."""
    
    def __init__(self, config: VADConfig) -> None:
        """
        Initialize VAD processor.
        
        Args:
            config: VAD configuration
        """
        self.config = config
        self.model: Optional[SileroVADModel] = None
        
        # State management
        self.is_voice_active = False
        self.voice_start_frame_count = 0
        self.voice_end_frame_count = 0
        self.voice_probabilities = deque(maxlen=100)  # Keep recent probabilities
        
        # Audio buffer for voice recording
        self.voice_buffer: deque = deque()
        self.current_voice_data: Optional[np.ndarray] = None
        
        # WAV writer for output
        self.wav_writer = WAVWriter(
            sample_rate=config.output_wav_sample_rate,
            bit_depth=config.output_wav_bit_depth,
            channels=1
        )
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Silero VAD model."""
        try:
            # Determine model path
            if self.config.model_path:
                model_dir = self.config.model_path
            else:
                # Use package resources
                model_dir = Path(__file__).parent.parent / "models"
            
            model_filename = self.config.get_model_filename()
            model_path = model_dir / model_filename
            
            if not model_path.exists():
                raise ModelNotFoundError(str(model_path))
            
            self.model = SileroVADModel(str(model_path), self.config.model_version)
            
        except Exception as e:
            raise ModelInitializationError(
                self.config.model_version.value,
                f"Failed to initialize VAD processor: {str(e)}"
            )
    
    def process_frame(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single audio frame.
        
        Args:
            audio_frame: Audio frame data (float32)
            
        Returns:
            Dictionary with processing results
        """
        try:
            if self.model is None:
                raise ModelInitializationError(self.config.model_version.value, "Model not loaded")
            
            # Validate and preprocess audio
            AudioUtils.validate_audio_data(audio_frame)
            
            # Apply denoising if enabled
            if self.config.enable_denoising:
                audio_frame = AudioUtils.denoise_audio(audio_frame)
            
            # Get voice probability from model
            probability = self.model.predict(audio_frame, self.config.sample_rate)
            self.voice_probabilities.append(probability)
            
            # Process voice activity state
            result = self._process_voice_state(probability, audio_frame)
            result['probability'] = probability
            
            return result
            
        except Exception as e:
            raise AudioProcessingError(f"Frame processing failed: {str(e)}")
    
    def _process_voice_state(self, probability: float, audio_frame: np.ndarray) -> Dict[str, Any]:
        """Process voice activity state machine."""
        result = {
            'voice_started': False,
            'voice_ended': False,
            'voice_continuing': False,
            'wav_data': None,
            'pcm_data': None
        }
        
        # Check for voice start
        if not self.is_voice_active:
            if probability >= self.config.vad_start_probability:
                self.voice_start_frame_count += 1
                
                # Add frame to potential voice buffer
                self.voice_buffer.append(audio_frame.copy())
                
                # Confirm voice start
                if self.voice_start_frame_count >= self.config.voice_start_frame_count:
                    self.is_voice_active = True
                    self.voice_start_frame_count = 0
                    self.voice_end_frame_count = 0
                    
                    # Initialize voice recording with buffered data
                    self.current_voice_data = np.concatenate(list(self.voice_buffer))
                    self.voice_buffer.clear()
                    
                    result['voice_started'] = True
            else:
                # Reset if probability drops
                self.voice_start_frame_count = 0
                self.voice_buffer.clear()
        
        # Check for voice end
        else:  # is_voice_active
            # Add frame to voice recording
            if self.current_voice_data is not None:
                self.current_voice_data = np.concatenate([self.current_voice_data, audio_frame])
            else:
                self.current_voice_data = audio_frame.copy()
            
            # Provide continuous PCM data
            result['voice_continuing'] = True
            result['pcm_data'] = audio_frame.tobytes()
            
            if probability < self.config.vad_end_probability:
                self.voice_end_frame_count += 1
                
                # Confirm voice end
                if self.voice_end_frame_count >= self.config.voice_end_frame_count:
                    self.is_voice_active = False
                    self.voice_end_frame_count = 0
                    
                    # Generate WAV data
                    if self.current_voice_data is not None:
                        wav_data = self.wav_writer.write_wav_data(self.current_voice_data)
                        result['wav_data'] = wav_data
                    
                    result['voice_ended'] = True
                    self.current_voice_data = None
            else:
                # Reset end count if probability rises
                self.voice_end_frame_count = 0
        
        return result
    
    def reset(self) -> None:
        """Reset VAD processor state."""
        if self.model:
            self.model.reset()
        
        self.is_voice_active = False
        self.voice_start_frame_count = 0
        self.voice_end_frame_count = 0
        self.voice_probabilities.clear()
        self.voice_buffer.clear()
        self.current_voice_data = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'is_voice_active': self.is_voice_active,
            'voice_start_frame_count': self.voice_start_frame_count,
            'voice_end_frame_count': self.voice_end_frame_count,
            'recent_probabilities': list(self.voice_probabilities),
            'average_probability': np.mean(self.voice_probabilities) if self.voice_probabilities else 0.0,
            'voice_buffer_size': len(self.voice_buffer),
            'current_voice_length': len(self.current_voice_data) if self.current_voice_data is not None else 0
        }
