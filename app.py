import streamlit as st
import streamlit.components.v1 as components
import librosa
import tempfile
import os
import json
import soundfile as sf
import numpy as np
import base64
import time
from scipy.signal import butter, sosfilt

# Konfigurasi Halaman
st.set_page_config(page_title="Chord Tracker by AIrawan", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Chord Tracker by AIrawan")

# --- HELPER FUNCTIONS ---

def apply_highpass_filter(y, sr, cutoff_freq=80):
    sos = butter(10, cutoff_freq, 'hp', fs=sr, output='sos')
    return sosfilt(sos, y)

def process_time_stretch(y, sr, rate_factor):
    """
    Time Stretch MURNI menggunakan Librosa (Tanpa dependensi luar).
    Ini solusi paling stabil untuk Streamlit Cloud.
    """
    if rate_factor == 1.0:
        return y
    
    # Melakukan STFT, Stretch dengan Phase Vocoder, lalu Inverse STFT
    # Ini murni Python, tidak butuh aplikasi eksternal rubberband
    stft = librosa.stft(y)
    stft_stretched = librosa.phase_vocoder(stft, rate=rate_factor)
    return librosa.istft(stft_stretched)

def reset_tempo():
    st.session_state.speed_slider = 1.0

# --- FUNGSI DETEKSI AKORD ---

def generate_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_types = {
        'maj': {'intervals': [0, 4, 7], 'weights': [1.0, 1.0, 0.8]},
        'min': {'intervals': [0, 3, 7], 'weights': [1.0, 1.0, 0.8]}
    }
    templates = []
    labels = []
    for i, root in enumerate(pitch_classes):
        for chord_name, data in chord_types.items():
            template = np.zeros(12)
            for interval, weight in zip(data['intervals'], data['weights']):
                template[(i + interval) % 12] = weight
            norm = np.linalg.norm(template)
            if norm > 0: template = template / norm
            label = f"{root}{'m' if chord_name == 'min' else ''}"
            templates.append(template)
            labels.append(label)
    return np.array(templates).T, labels

def detect_chords_librosa(y, sr, beats):
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), bins_per_octave=36)
    if len(beats) > 0:
        chroma_sync = librosa.util.sync(chroma_cens, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(beats, sr=sr)
    else:
        chroma_sync = chroma_cens
        beat_times = librosa.times_like(chroma_cens, sr=sr)
        
    templates, labels = generate_chord_templates()
    similarities = np.dot(templates.T, chroma_sync)
    best_matches = np.argmax(similarities, axis=0)
    
    chords = []
    current_chord = None
    for time_val, match_idx in zip(beat_times, best_matches):
        chord_label = labels[match_idx]
        if chord_label != current_chord:
            chords.append({"time": float(time_val), "label": chord_label})
            current_chord = chord_label
    return chords

# --- MAIN APP ---

uploaded_file = st.file_uploader("Unggah file audio (WAV / MP3)", type=["wav", "mp3"])

if uploaded_file:
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name
        
        data, rate = librosa.load(path, sr=44100, mono=True)
        tempo, beats = librosa.beat.beat_track(y=data, sr=rate)
        
        st.session_state.detected_bpm = int(round(float(np.mean(tempo))))
        st.session_state.raw_chords = detect_chords_librosa(librosa.effects.hpss(apply_highpass_filter(data, rate))[0], rate, beats)
        st.session_state.y_full, st.session_state.sr_full = librosa.load(path, sr=44100, mono=False)
        st.session_state.file_id = str(time.time())

    # UI Controls
    speed = st.slider("Kecepatan", 0.5, 1.5, 1.0, 0.05, key="speed_slider")
    
    # Process Audio (Tanpa PyRubberband sama sekali)
    processed_path = os.path.join(tempfile.gettempdir(), f"out_{st.session_state.file_id}.wav")
    
    if not os.path.exists(processed_path):
        y = st.session_state.y_full
        sr = st.session_state.sr_full
        if y.ndim == 2:
            y_out = np.vstack([process_time_stretch(y[0], sr, speed), process_time_stretch(y[1], sr, speed)]).T
        else:
            y_out = process_time_stretch(y, sr, speed)
        sf.write(processed_path, y_out, sr)

    # Encode & Player
    with open(processed_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    chords_json = json.dumps([{"time": c["time"] / speed, "label": c["label"]} for c in st.session_state.raw_chords])

    # Player HTML
    html_code = f"""
    <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
    <div id="waveform"></div>
    <script>
        const ws = WaveSurfer.create({{ container: '#waveform', waveColor: '#30363d', progressColor: '#58a6ff', height: 100 }});
        ws.load('data:audio/wav;base64,{audio_b64}');
    </script>
    """
    components.html(html_code, height=150)
