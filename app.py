import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import base64
import tempfile
import json

st.set_page_config(page_title="Accurate Chord Tracker", layout="wide")

st.title("🎸 Accurate Chord Tracker")
st.markdown("Deteksi akord otomatis berbasis CENS Chromagram & Beat-Sync.")

# --- Helper: Chord Template ---
def get_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Chord template sederhana
    chord_types = {
        'maj': [1,0,0,0,1,0,0,1,0,0,0,0],
        'min': [1,0,0,1,0,0,0,1,0,0,0,0]
    }
    templates = []
    labels = []
    for i, root in enumerate(pitch_classes):
        for name, pattern in chord_types.items():
            # Rotasi pattern sesuai root
            template = np.roll(pattern, i)
            templates.append(template)
            labels.append(f"{root}{name if name == 'maj' else 'm'}")
    return np.array(templates), labels

# --- Core Detection ---
def detect_chords(y, sr):
    # 1. High-quality Chroma CENS (Energy Normalized)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'))
    
    # 2. Beat Tracking untuk sinkronisasi akord (agar lebih stabil)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # 3. Matching dengan template
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Menganalisis audio..."):
        y, sr = librosa.load(path, sr=22050)
        chords = detect_chords(y, sr)
        
        # Encode audio ke base64 untuk pemutar JS
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.subheader("Hasil Deteksi")
    st.write(chords)

    # --- WaveSurfer JS Player ---
    player_html = f"""
    <div id="waveform" style="background:#1e1e1e; padding:10px; border-radius:8px;"></div>
    <div id="chord-display" style="font-size:2em; font-weight:bold; margin-top:10px; color:#58a6ff;">-</div>
    <script src="https://unpkg.com/wavesurfer.js"></script>
    <script>
        var wavesurfer = WaveSurfer.create({{container: '#waveform', waveColor: '#58a6ff', progressColor: '#f0883e', height: 100}});
        wavesurfer.load('data:audio/wav;base64,{audio_b64}');
        var chords = {json.dumps(chords)};
        wavesurfer.on('audioprocess', function() {{
            var time = wavesurfer.getCurrentTime();
            var current = chords.filter(c => c.time <= time).pop();
            if(current) document.getElementById('chord-display').innerText = current.label;
        }});
    </script>
    """
    components.html(player_html, height=250)
