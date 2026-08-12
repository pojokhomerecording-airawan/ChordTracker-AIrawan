# Berikut adalah script Streamlit yang telah diperbarui dengan **menghapus seluruh fitur *time-stretching*** (pengubahan kecepatan tempo), sehingga pemrosesan audio menjadi jauh lebih cepat dan ringan:

python
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
        .stMetric { background-color: #161b22; padding: 10px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Chord Tracker by AIrawan")

# --- HELPER FUNCTIONS ---

def apply_highpass_filter(y, sr, cutoff_freq=80):
    """High-pass filter menggunakan SOS"""
    sos = butter(10, cutoff_freq, 'hp', fs=sr, output='sos')
    return sosfilt(sos, y)

# --- FUNGSI DETEKSI AKORD PRESISI (CENS + BEAT-SYNC) ---

def generate_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_types = {
        'maj':   {'intervals': [0, 4, 7],        'weights': [1.0, 1.0, 0.8]},
        'min':   {'intervals': [0, 3, 7],        'weights': [1.0, 1.0, 0.8]},
        'dim':   {'intervals': [0, 3, 6],        'weights': [1.0, 1.0, 0.8]},
        'aug':   {'intervals': [0, 4, 8],        'weights': [1.0, 1.0, 0.8]},
        'maj7':  {'intervals': [0, 4, 7, 11],    'weights': [1.0, 1.0, 0.8, 0.9]},
        'm7':    {'intervals': [0, 3, 7, 10],    'weights': [1.0, 1.0, 0.8, 0.9]},
        '7':     {'intervals': [0, 4, 7, 10],    'weights': [1.0, 1.0, 0.8, 0.9]},
        'dim7':  {'intervals': [0, 3, 6, 9],     'weights': [1.0, 1.0, 0.8, 0.9]},
        'm7b5':  {'intervals': [0, 3, 6, 10],    'weights': [1.0, 1.0, 0.8, 0.9]}
    }
    templates = []
    labels = []
    for i, root in enumerate(pitch_classes):
        for chord_name, data in chord_types.items():
            template = np.zeros(12)
            for interval, weight in zip(data['intervals'], data['weights']):
                template[(i + interval) % 12] = weight
            norm = np.linalg.norm(template)
            if norm > 0:
                template = template / norm
            if chord_name == 'maj': label = root
            elif chord_name == 'min': label = f"{root}m"
            else: label = f"{root}{chord_name}"
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
    
    for idx, label in enumerate(labels):
        if any(x in label for x in ['7', 'maj7', 'm7', 'dim7', 'm7b5']):
            similarities[idx, :] *= 0.95
            
    best_matches = np.argmax(similarities, axis=0)
    
    chords = []
    current_chord = None
    if len(beat_times) > 0 and beat_times[0] > 0.1:
        chords.append({"time": 0.0, "label": "N"})
        
    for time_val, match_idx in zip(beat_times, best_matches):
        chord_label = labels[match_idx]
        if chord_label != current_chord:
            chords.append({"time": float(time_val), "label": chord_label})
            current_chord = chord_label
            
    return chords

# --- MAIN APP ---

uploaded_file = st.file_uploader("Unggah file audio (WAV / MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    if "uploaded_name" not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
        st.session_state.uploaded_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.audio_path = tmp.name

        with st.spinner("Menganalisis BPM & Akord..."):
            data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
            tempo, beats = librosa.beat.beat_track(y=data, sr=rate)
            st.session_state.detected_bpm = int(round(float(np.mean(tempo))))

            data_hp = apply_highpass_filter(data, rate, cutoff_freq=80)
            data_harmonic, _ = librosa.effects.hpss(data_hp)
            
            st.session_state.raw_chords = detect_chords_librosa(data_harmonic, rate, beats)

    st.subheader("⚙️ Informasi Audio")
    st.markdown(f"**Auto BPM Original:** `{st.session_state.detected_bpm} BPM`")

    chords_json = json.dumps(st.session_state.raw_chords)

    with open(st.session_state.audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # --- HTML / JS WAVESURFER PLAYER ---
    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/plugin/wavesurfer.regions.min.js"></script>
        <style>
            body {{ background-color: #0d1117; color: white; font-family: -apple-system, sans-serif; margin: 0; padding: 10px; }}
            .player-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; }}
            #waveform {{ width: 100%; margin-bottom: 15px; position: relative; }}
            .controls {{ display: flex; align-items: center; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }}
            button {{ background-color: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; cursor: pointer; }}
            button:hover {{ background-color: #2ea043; }}
            .btn-loop-active {{ background-color: #d29922; }}
            .btn-clear {{ background-color: #21262d; border: 1px solid #30363d; }}
            .chord-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
            .chord-val {{ font-size: 56px; font-weight: bold; color: #58a6ff; margin-top: 5px; }}
            .chord-label-tag {{ position: absolute; top: 2px; left: 3px; background: rgba(31, 111, 235, 0.85); color: #ffffff; font-size: 10px; font-weight: bold; padding: 1px 4px; border-radius: 3px; pointer-events: none; }}
        </style>
    </head>
    <body>
        <div class="player-container">
            <div id="waveform"></div>
            <div class="controls">
                <button onclick="wavesurfer.playPause()">Play / Pause</button>
                <button id="btnLoop" onclick="toggleLoopMode()">🔁 Loop Section: OFF</button>
                <button class="btn-clear" onclick="clearCustomLoop()">❌ Hapus Seleksi</button>
            </div>
            <div class="chord-box">
                <div style="color: #8b949e; font-size: 11px; text-transform: uppercase;">Akord Aktif</div>
                <div id="chordDisplay" class="chord-val">-</div>
            </div>
        </div>
        <script>
            const chordData = {chords_json};
            const wavesurfer = WaveSurfer.create({{ container: '#waveform', waveColor: '#30363d', progressColor: '#58a6ff', cursorColor: '#f0883e', height: 90, plugins: [WaveSurfer.regions.create({{ dragSelection: true }})] }});
            wavesurfer.load('data:audio/wav;base64,{audio_b64}');
            
            wavesurfer.on('ready', () => {{
                chordData.forEach((item, index) => {{
                    const nextTime = (index < chordData.length - 1) ? chordData[index + 1].time : wavesurfer.getDuration();
                    const region = wavesurfer.addRegion({{ start: item.time, end: nextTime, drag: false, resize: false }});
                    if (region.element) {{
                        const tag = document.createElement('span'); tag.className = 'chord-label-tag'; tag.innerText = item.label; region.element.appendChild(tag);
                    }}
                }});
            }});

            let isLoopEnabled = false; let activeLoopRegion = null;
            function toggleLoopMode() {{ isLoopEnabled = !isLoopEnabled; document.getElementById('btnLoop').innerText = isLoopEnabled ? "🔁 Loop Section: ON" : "🔁 Loop Section: OFF"; document.getElementById('btnLoop').classList.toggle('btn-loop-active'); }}
            function clearCustomLoop() {{ if (activeLoopRegion) {{ activeLoopRegion.remove(); activeLoopRegion = null; }} }}
            wavesurfer.on('region-out', (r) => {{ if (isLoopEnabled && activeLoopRegion === r) {{ wavesurfer.seekTo(r.start / wavesurfer.getDuration()); wavesurfer.play(); }} }});
            wavesurfer.on('region-created', (r) => {{ if (r.drag || r.resize) {{ if (activeLoopRegion && activeLoopRegion !== r) activeLoopRegion.remove(); activeLoopRegion = r; if (!isLoopEnabled) toggleLoopMode(); }} }});
            wavesurfer.on('audioprocess', () => {{ const t = wavesurfer.getCurrentTime(); let c = "-"; for(let i=0; i<chordData.length; i++) {{ if(t >= chordData[i].time && (i === chordData.length-1 || t < chordData[i+1].time)) {{ c = chordData[i].label; break; }} }} document.getElementById('chordDisplay').innerText = c; }});
        </script>
    </body>
    </html>
    """
    components.html(player_html, height=350)

```
