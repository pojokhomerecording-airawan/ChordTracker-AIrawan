import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import tempfile
import json
import base64
import os

st.set_page_config(page_title="ChordTracker by AIrawan", layout="wide")

st.title("ChordTracker by AIrawan")
st.markdown("Deteksi akord otomatis")

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

# --- Core Detection (Optimized for Stability) ---
def detect_chords(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), hop_length=1024)
    chroma = librosa.util.normalize(chroma, norm=np.inf, axis=0)
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024)
    
    if len(beats) > 0:
        fixed_beats = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1])
        chroma_sync = librosa.util.sync(chroma, fixed_beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(fixed_beats, sr=sr, hop_length=1024)
    else:
        chroma_sync = chroma
        beat_times = librosa.times_like(chroma, sr=sr, hop_length=1024)
        
    templates, labels = get_chord_templates()
    similarities = np.dot(templates, chroma_sync)
    best_indices = np.argmax(similarities, axis=0)
    
    chord_sequence = []
    current_chord = None
    min_duration = 0.6  
    
    for i, idx in enumerate(best_indices):
        chord = labels[idx]
        t = float(beat_times[i])
        
        if chord != current_chord:
            if not chord_sequence or (t - chord_sequence[-1]["time"] >= min_duration):
                chord_sequence.append({"time": t, "label": chord})
                current_chord = chord
            else:
                chord_sequence[-1]["label"] = chord
                current_chord = chord
            
    return chord_sequence

# --- Main App ---
uploaded_file = st.file_uploader("Upload lagu (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file:
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        if "audio_path" in st.session_state and os.path.exists(st.session_state.audio_path):
            try:
                os.remove(st.session_state.audio_path)
            except:
                pass
                
        st.session_state.clear()
        st.session_state.file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.audio_path = tmp.name

        with st.spinner("Menganalisis audio & akord lagu baru..."):
            y, sr = librosa.load(st.session_state.audio_path, sr=22050)
            st.session_state.chords = detect_chords(y, sr)

    chords_json = json.dumps(st.session_state.chords)

    with open(st.session_state.audio_path, "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # --- HTML / JS WAVESURFER PLAYER DENGAN SEEKING AKTIF ---
    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
        <style>
            body {{ background-color: #0e1117; color: white; font-family: sans-serif; margin: 0; padding: 10px; }}
            .player-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
            #waveform {{ width: 100%; margin-bottom: 15px; cursor: pointer; }}
            .controls {{ display: flex; gap: 10px; margin-bottom: 20px; }}
            button {{ background-color: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px; }}
            button:hover {{ background-color: #2ea043; }}
            .chord-container {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
            .chord-title {{ color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }}
            .chord-display {{ font-size: 64px; font-weight: bold; color: #58a6ff; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="player-box">
            <div id="waveform"></div>
            <div class="controls">
                <button onclick="wavesurfer.playPause()">▶ Play / Pause</button>
            </div>
            <div class="chord-container">
                <div class="chord-title">Akord Aktif</div>
                <div id="chordDisplay" class="chord-display">-</div>
            </div>
        </div>

        <script>
            const chordData = {chords_json};

            const wavesurfer = WaveSurfer.create({{
                container: '#waveform',
                waveColor: '#30363d',
                progressColor: '#58a6ff',
                cursorColor: '#f0883e',
                cursorWidth: 3,
                height: 100,
                barWidth: 2,
                barRadius: 2,
                interact: true,  // Memastikan interaksi klik/geser aktif
                responsive: true
            }});

            const base64Audio = "{audio_b64}";
            const binaryString = atob(base64Audio);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            const blob = new Blob([bytes], {{ type: 'audio/wav' }});
            const blobUrl = URL.createObjectURL(blob);
            
            wavesurfer.load(blobUrl);

            // Fungsi helper untuk memperbarui tampilan akor berdasarkan waktu tertentu
            function updateChordDisplay(time) {{
                let currentChord = "-";
                for (let i = 0; i < chordData.length; i++) {{
                    if (time >= chordData[i].time) {{
                        currentChord = chordData[i].label;
                    }} else {{
                        break;
                    }}
                }}
                document.getElementById('chordDisplay').innerText = currentChord;
            }}

            // Sinkronisasi saat lagu berjalan normal
            wavesurfer.on('audioprocess', () => {{
                updateChordDisplay(wavesurfer.getCurrentTime());
            }});

            // Sinkronisasi instan saat user mengklik atau menggeser playhead ke posisi baru
            wavesurfer.on('seek', (progress) => {{
                updateChordDisplay(wavesurfer.getCurrentTime());
            }});
        </script>
    </body>
    </html>
    """

    components.html(player_html, height=350)
