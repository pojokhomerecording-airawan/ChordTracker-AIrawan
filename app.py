import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import tempfile
import json
import base64

st.set_page_config(page_title="Accurate Chord Tracker", layout="wide")

st.title("🎸 Accurate Chord Tracker with Waveform")
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
    if "audio_path" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.audio_path = tmp.name
            st.session_state.file_name = uploaded_file.name

        with st.spinner("Menganalisis audio & akord..."):
            y, sr = librosa.load(st.session_state.audio_path, sr=22050)
            st.session_state.chords = detect_chords(y, sr)

    chords_json = json.dumps(st.session_state.chords)

    # Membaca file audio ke base64 untuk dimasukkan aman ke JavaScript blob
    with open(st.session_state.audio_path, "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # --- HTML / JS WAVESURFER PLAYER DENGAN PLAYHEAD ---
    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
        <style>
            body {{ background-color: #0e1117; color: white; font-family: sans-serif; margin: 0; padding: 10px; }}
            .player-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
            #waveform {{ width: 100%; margin-bottom: 15px; }}
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

            // Inisialisasi WaveSurfer dengan Playhead (cursorColor)
            const wavesurfer = WaveSurfer.create({{
                container: '#waveform',
                waveColor: '#30363d',
                progressColor: '#58a6ff',
                cursorColor: '#f0883e',  // Garis playhead berjalan
                cursorWidth: 2,
                height: 100,
                barWidth: 2,
                barRadius: 2,
                responsive: true
            }});

            // Load audio dari base64 aman via Blob
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

            // Sinkronisasi pemutaran akord real-time berdasarkan waktu playhead
            wavesurfer.on('audioprocess', () => {{
                const currentTime = wavesurfer.getCurrentTime();
                let currentChord = "-";
                
                for (let i = 0; i < chordData.length; i++) {{
                    if (currentTime >= chordData[i].time) {{
                        currentChord = chordData[i].label;
                    }} else {{
                        break;
                    }}
                }}
                document.getElementById('chordDisplay').innerText = currentChord;
            }});

            // Update juga ketika user melakukan klik/seek langsung pada waveform
            wavesurfer.on('seek', () => {{
                const currentTime = wavesurfer.getCurrentTime();
                let currentChord = "-";
                
                for (let i = 0; i < chordData.length; i++) {{
                    if (currentTime >= chordData[i].time) {{
                        currentChord = chordData[i].label;
                    }} else {{
                        break;
                    }}
                }}
                document.getElementById('chordDisplay').innerText = currentChord;
            }});
        </script>
    </body>
    </html>
    """

    components.html(player_html, height=290)

    st.subheader("📋 Data Timeline Akord")
    st.write(st.session_state.chords)
