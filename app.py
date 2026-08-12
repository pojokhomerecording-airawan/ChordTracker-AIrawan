import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import tempfile
import json
import os

st.set_page_config(page_title="Accurate Chord Tracker", layout="wide")

st.title("🎸 Accurate Chord Tracker")
st.markdown("Deteksi akord otomatis berbasis CENS Chromagram & Beat-Sync.")

# --- Helper: Chord Template ---
def get_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_types = {
        'maj': [1,0,0,0,1,0,0,1,0,0,0,0],
        'min': [1,0,0,1,0,0,0,1,0,0,0,0]
    }
    templates = []
    labels = []
    for i, root in enumerate(pitch_classes):
        for name, pattern in chord_types.items():
            template = np.roll(pattern, i)
            templates.append(template)
            labels.append(f"{root}{name if name == 'maj' else 'm'}")
    return np.array(templates), labels

# --- Core Detection ---
def detect_chords(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    if len(beats) > 0:
        # Menggunakan safe frame handling
        fixed_beats = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1])
        chroma_sync = librosa.util.sync(chroma, fixed_beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(fixed_beats, sr=sr)
    else:
        chroma_sync = chroma
        beat_times = librosa.times_like(chroma, sr=sr)
        
    templates, labels = get_chord_templates()
    similarities = np.dot(templates, chroma_sync)
    best_indices = np.argmax(similarities, axis=0)
    
    chord_sequence = []
    current_chord = None
    
    for i, idx in enumerate(best_indices):
        chord = labels[idx]
        if chord != current_chord:
            chord_sequence.append({"time": float(beat_times[i]), "label": chord})
            current_chord = chord
            
    return chord_sequence

# --- Main App ---
uploaded_file = st.file_uploader("Upload lagu (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file:
    # Simpan ke temp file permanen selama sesi aktif
    if "audio_path" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.audio_path = tmp.name
            st.session_state.file_name = uploaded_file.name

        with st.spinner("Menganalisis audio & akord..."):
            y, sr = librosa.load(st.session_state.audio_path, sr=22050)
            st.session_state.chords = detect_chords(y, sr)

    st.subheader("🔊 Pemutar Audio & Kontrol")
    
    # Menggunakan st.audio bawaan Streamlit (Sangat stabil untuk semua jenis file MP3/WAV)
    st.audio(st.session_state.audio_path)

    st.subheader("Hasil Deteksi Akord (Timeline)")
    st.write(st.session_state.chords)
