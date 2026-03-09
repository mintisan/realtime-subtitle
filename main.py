import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import signal
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from audio_capture import AudioCapture
from transcriber import Transcriber
from translator import Translator
from overlay_window import OverlayWindow
from config import config

class WorkerSignals(QObject):
    update_text = pyqtSignal(int, str, str)  # (chunk_id, original, translated)

class Pipeline(QObject):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True
        self.state_lock = threading.Lock()
        self.partial_future = None
        self.last_final_text = ""
        self.active_chunk_id = 1
        self.latest_partial_request = 0
        self.translation_enabled = True
        
        # Print config for debugging
        config.print_config()
        
        # Initialize components
        self.audio = AudioCapture(
            device_index=config.device_index,
            sample_rate=config.sample_rate,
            silence_threshold=config.silence_threshold,
            silence_duration=config.silence_duration,
            chunk_duration=config.chunk_duration,
            max_phrase_duration=config.max_phrase_duration,
            streaming_mode=config.streaming_mode,
            streaming_interval=config.streaming_interval,
            streaming_step_size=config.streaming_step_size,
            streaming_overlap=config.streaming_overlap
        )
        
        # Initialize Transcriber
        print(f"[Pipeline] Initializing Transcriber with backend={config.asr_backend}, device={config.whisper_device}...")
        
        # Determine model size based on backend
        if config.asr_backend == "funasr":
            model_size = config.funasr_model
        else:
            model_size = config.whisper_model
            
        self.transcriber = Transcriber(
            backend=config.asr_backend,
            model_size=model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            language=config.source_language
        )
        self.live_transcriber = self.transcriber
        self.has_dedicated_live_transcriber = False
        
        # Keep live partial updates responsive by using a separate Whisper instance.
        if config.asr_backend == "whisper":
            try:
                print("[Pipeline] Initializing dedicated live transcriber for partial updates...")
                self.live_transcriber = Transcriber(
                    backend=config.asr_backend,
                    model_size=model_size,
                    device=config.whisper_device,
                    compute_type=config.whisper_compute_type,
                    language=config.source_language
                )
                self.has_dedicated_live_transcriber = True
            except Exception as e:
                print(f"[Pipeline] Live transcriber unavailable, reusing final transcriber: {e}")
                self.live_transcriber = self.transcriber
        
        # Initialize Translator
        print(f"[Pipeline] Initializing Translator (target={config.target_lang})...")
        self.translator = Translator(
            target_lang=config.target_lang,
            base_url=config.api_base_url,
            api_key=config.api_key,
            model=config.model
        )
        
        # Warmup Transcriber (Critical for MLX/GPU)
        self.transcriber.warmup()
        if self.live_transcriber is not self.transcriber:
            self.live_transcriber.warmup()

    def start(self):
        """Start the processing pipeline in a dedicated thread"""
        # self.audio.start() # DISABLE: Generator manages its own stream. calling this causes double-stream error on macOS
        self.thread = threading.Thread(target=self.processing_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        print("\n[Pipeline] Stopping...")
        self.running = False
        self.audio.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        print("[Pipeline] Stopped.")

    def set_translation_enabled(self, enabled):
        with self.state_lock:
            self.translation_enabled = enabled
        print(f"[Pipeline] Translation {'enabled' if enabled else 'disabled'}")

    def processing_loop(self):
        """Fully parallel pipeline: multiple concurrent transcription + translation"""
        print("Pipeline processing loop started.")
        if self.has_dedicated_live_transcriber:
            print("[Pipeline] Using a dedicated live transcriber for partial updates.")
        
        import numpy as np
        
        # Executors
        partial_executor = ThreadPoolExecutor(max_workers=1)
        final_executor = partial_executor
        if self.live_transcriber is not self.transcriber:
            final_executor = ThreadPoolExecutor(max_workers=1)
        translate_executor = ThreadPoolExecutor(max_workers=config.translation_threads)
        
        # State
        buffer = np.array([], dtype=np.float32)
        chunk_id = 1
        last_update_time = time.time()
        phrase_start_time = None
        partial_updates_for_chunk = 0
        
        # Generator yielding small chunks (e.g. 0.2s)
        audio_gen = self.audio.generator()
        
        overlap_samples = int(self.audio.sample_rate * config.final_overlap_duration)

        try:
            for audio_chunk in audio_gen:
                if not self.running:
                    break
                buffer = np.concatenate([buffer, audio_chunk])
                now = time.time()
                buffer_duration = len(buffer) / self.audio.sample_rate
                if phrase_start_time is None:
                    phrase_start_time = now
                chunk_wall_duration = now - phrase_start_time
                
                # Check silence for finalization
                # Use configured silence duration/threshold
                is_silence = False
                min_silence_dur = max(config.silence_duration, 0.1)
                
                # Only check silence if we have enough buffer
                if buffer_duration > min_silence_dur:
                    # Check tail of silence duration
                    tail_samples = int(self.audio.sample_rate * min_silence_dur)
                    tail = buffer[-tail_samples:]
                    rms = np.sqrt(np.mean(tail**2))
                    if rms < self.audio.silence_threshold:
                        is_silence = True
                        
                # Dynamic VAD Logic
                # 1. Standard: > 2.0s duration AND > 1.0s silence (Configured)
                standard_cut = (is_silence and buffer_duration > 2.0)
                
                # 2. Soft Limit: > 6.0s duration AND > 0.4s silence (Catch brief pauses to avoid huge latency)
                soft_limit_cut = False
                if buffer_duration > 6.0 or chunk_wall_duration > 6.0:
                    # Check shorter silence tail (0.4s)
                    short_tail_samps = int(self.audio.sample_rate * 0.4)
                    if len(buffer) > short_tail_samps:
                        t_rms = np.sqrt(np.mean(buffer[-short_tail_samps:]**2))
                        if t_rms < self.audio.silence_threshold:
                            soft_limit_cut = True
                            
                # 3. Hard Limit: > max_phrase_duration in audio length or wall clock time
                hard_limit_cut = (
                    buffer_duration > self.audio.max_phrase_duration
                    or (
                        buffer_duration > 2.0
                        and chunk_wall_duration > (self.audio.max_phrase_duration + config.update_interval)
                    )
                )

                should_finalize = standard_cut or soft_limit_cut or hard_limit_cut
                
                if should_finalize and buffer_duration > 0.5:
                    # FINALIZE
                    final_buffer = buffer.copy()
                    next_buffer = np.array([], dtype=np.float32)
                    if overlap_samples > 0 and len(final_buffer) > overlap_samples:
                        next_buffer = final_buffer[-overlap_samples:].copy()
                    cid = chunk_id
                    
                    # Store current prompt to pass to task (thread safety)
                    prompt = self.last_final_text
                    
                    # PRE-CHECK: Is the entire buffer actually silence?
                    # (Prevent infinite loop of repeating prompt on empty audio)
                    overall_rms = np.sqrt(np.mean(final_buffer**2))
                    if overall_rms < self.audio.silence_threshold:
                        print(
                            f"[Pipeline] Skipped silent chunk {cid} "
                            f"(RMS={overall_rms:.4f}, wall={chunk_wall_duration:.2f}s, buffer={buffer_duration:.2f}s)"
                        )
                    else:
                        reasons = []
                        if standard_cut:
                            reasons.append("silence")
                        if soft_limit_cut:
                            reasons.append("soft_limit")
                        if hard_limit_cut:
                            reasons.append("hard_limit")
                        print(
                            f"[Pipeline] Finalizing chunk {cid} "
                            f"(reason={'+'.join(reasons)}, wall={chunk_wall_duration:.2f}s, "
                            f"buffer={buffer_duration:.2f}s, partials={partial_updates_for_chunk})"
                        )
                        # Submit Final Task
                        # Pass prompt AND translate_executor for async translation
                        final_executor.submit(self._process_final_chunk, final_buffer, cid, prompt, translate_executor)
                    
                    # Reset
                    buffer = next_buffer
                    chunk_id += 1
                    partial_updates_for_chunk = 0
                    with self.state_lock:
                        self.active_chunk_id = chunk_id
                        self.latest_partial_request += 1
                    phrase_start_time = now if len(next_buffer) > 0 else None
                    last_update_time = now
                    
                # 2. Partial Update if: Interval passed AND not finalizing
                elif now - last_update_time > config.update_interval and buffer_duration > 0.5:
                    # PARTIAL UPDATE
                    partial_buffer = buffer.copy()
                    prompt = self.last_final_text
                    
                    # RMS Check to avoid partial hallucination on silence
                    rms = np.sqrt(np.mean(partial_buffer**2))
                    if rms > self.audio.silence_threshold and (self.partial_future is None or self.partial_future.done()):
                        with self.state_lock:
                            self.active_chunk_id = chunk_id
                            self.latest_partial_request += 1
                            request_id = self.latest_partial_request
                        partial_updates_for_chunk += 1
                        self.partial_future = partial_executor.submit(
                            self._process_partial_chunk, partial_buffer, chunk_id, prompt, request_id
                        )
                    
                    last_update_time = now
                    
        except Exception as e:
            print(f"[Pipeline] Error in loop: {e}")
        finally:
            partial_executor.shutdown(wait=False)
            if final_executor is not partial_executor:
                final_executor.shutdown(wait=False)
            translate_executor.shutdown(wait=False)

    def _process_partial_chunk(self, audio_data, chunk_id, prompt="", request_id=0):
        """Transcribe and update UI (No translation)"""
        try:
            # Use accumulated context as prompt
            text = self.live_transcriber.transcribe(audio_data, prompt=prompt)
            with self.state_lock:
                is_stale = request_id != self.latest_partial_request or chunk_id != self.active_chunk_id
            if is_stale:
                return
            if text:
                self.signals.update_text.emit(chunk_id, text, "")
        except Exception as e:
            print(f"[Partial {chunk_id}] Error: {e}")

    def _process_final_chunk(self, audio_data, chunk_id, prompt="", translate_executor=None):
        """Transcribe, Log, and Trigger Translation Async"""
        try:
            text = self.transcriber.transcribe(audio_data, prompt=prompt)
            if text:
                print(f"[Final {chunk_id}] Transcribed: {text}")
                # Save for context (only if meaningful)
                if len(text.split()) > 2:
                    self.last_final_text = self._trim_prompt_context(text)

                with self.state_lock:
                    translation_enabled = self.translation_enabled

                # Emit final transcription first (confirms text)
                if translation_enabled:
                    self.signals.update_text.emit(chunk_id, text, "(translating...)")
                else:
                    self.signals.update_text.emit(chunk_id, text, "")

                # Offload translation to separate thread so we don't block next transcription
                if translation_enabled and translate_executor:
                    translate_executor.submit(self._run_translation, text, chunk_id)
            else:
                pass
        except Exception as e:
            print(f"[Final {chunk_id}] Error: {e}")

    def _run_translation(self, text, chunk_id):
        """Run translation in background and emit result"""
        started_at = time.time()
        print(f"[Translation {chunk_id}] Started")
        try:
            with self.state_lock:
                if not self.translation_enabled:
                    print(f"[Translation {chunk_id}] Skipped because translation is disabled")
                    return
            translated = self.translator.translate(text)
            elapsed = time.time() - started_at
            with self.state_lock:
                if not self.translation_enabled:
                    print(f"[Translation {chunk_id}] Dropped result because translation is disabled")
                    return
            print(f"[Final {chunk_id}] Translated in {elapsed:.2f}s: {translated}")
            self.signals.update_text.emit(chunk_id, text, translated)
        except Exception as e:
            elapsed = time.time() - started_at
            print(f"[Translation {chunk_id}] Failed after {elapsed:.2f}s: {e}")
            self.signals.update_text.emit(chunk_id, text, "[Translation Failed]")

    def _trim_prompt_context(self, text, max_words=20):
        """Keep prompt context short so carry-over helps continuity without over-biasing."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[-max_words:])
    
    def _transcribe_chunk(self, transcriber, audio_chunk, chunk_id):
        """Transcribe a single chunk and log timing"""
        t0 = time.time()
        text = transcriber.transcribe(audio_chunk)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Transcribed in {t1-t0:.2f}s: {text if text else '(empty)'}")
        return text
    
    def _translate_and_log(self, text, chunk_id=0):
        """Translate text and log result"""
        t0 = time.time()
        translated_text = self.translator.translate(text)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Translated in {t1-t0:.2f}s: {translated_text}")
        return (text, translated_text)

# Global reference for signal handler
_pipeline = None
_app = None

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n[Main] Ctrl-C received, force killing...")
    os._exit(0)

def start_overlay_session():
    """Start the overlay and pipeline without blocking (for use in Dashboard)"""
    global _pipeline, _app
    
    # Initialize Overlay Window
    window = OverlayWindow(
        display_duration=config.display_duration,
        window_width=config.window_width,
        window_height=config.window_height,
    )
    window.show()
    
    # Logic
    _pipeline = Pipeline()
    
    # Connect signals
    _pipeline.signals.update_text.connect(window.update_text)
    if hasattr(window, "bilingual_toggled"):
        window.bilingual_toggled.connect(_pipeline.set_translation_enabled)
    
    # Start pipeline
    _pipeline.start()
    
    return window, _pipeline

def main():
    global _pipeline, _app
    
    # Set up signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    _app = QApplication.instance()
    if not _app:
        _app = QApplication(sys.argv)
    
    # Start session
    win, pipe = start_overlay_session()
    
    # Timer to let Python interpreter handle signals (Ctrl-C)
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)
    
    try:
        sys.exit(_app.exec())
    except SystemExit:
        pass
    finally:
        if _pipeline:
            _pipeline.stop()

if __name__ == "__main__":
    main()
