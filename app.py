python
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

# --- Core Detection (Fixed Indexing) ---
def detect_chords(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    if len(beats) > 0:
        # Perbaikan: Menggunakan fix_frames agar sinkronisasi aman dari error dimensi
        fixed_beats = librosa.util.fix_frames(beats, xmin=0, xmax=chroma.shape[1])
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Menganalisis audio..."):
        y, sr = librosa.load(path, sr=22050)
        chords = detect_chords(y, sr)
        
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    st.subheader("Hasil Deteksi Akord")
    st.write(chords)

    # --- WaveSurfer JS Player ---
    player_html = f"""
    <div style="background:#161b22; padding:15px; border-radius:8px; border:1px solid #30363d;">
        <div id="waveform" style="margin-bottom:10px;"></div>
        <div style="color:#8b949e; font-size:12px; text-transform:uppercase;">Akord Aktif</div>
        <div id="chord-display" style="font-size:48px; font-weight:bold; color:#58a6ff; margin-top:5px;">-</div>
    </div>
    
    <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
    <script>
        const chords = {json.dumps(chords)};
        const wavesurfer = WaveSurfer.create({{
            container: '#waveform',
            waveColor: '#30363d',
            progressColor: '#58a6ff',
            cursorColor: '#f0883e',
            height: 90
        }});
        
        wavesurfer.load('data:audio/wav;base64,{audio_b64}');
        
        wavesurfer.on('audioprocess', function() {{
            const time = wavesurfer.getCurrentTime();
            let current = "-";
            for (let i = 0; i < chords.length; i++) {{
                if (time >= chords[i].time) {{
                    current = chords[i].label;
                }} else {{
                    break;
                }}
            }}
            document.getElementById('chord-display').innerText = current;
        }});
    </script>
    """
    components.html(player_html, height=240)

```
