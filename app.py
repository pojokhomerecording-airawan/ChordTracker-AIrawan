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

# Opsional: Gunakan pyrubberband jika tersedia untuk time-stretch kualitas tinggi
try:
    import pyrubberband as pyrb
    USE_RUBBERBAND = True
except ImportError:
    USE_RUBBERBAND = False

# Konfigurasi Halaman
st.set_page_config(page_title="Chord Tracker by AIrawan", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        .stMetric { background-color: #161b22; padding: 10px; border-radius: 10px; border: 1px solid #30363d; }
        
        /* CSS khusus hanya untuk tombol reset dengan key btn_reset_text */
        div[data-testid="stButton"] button[kind="secondary"] p,
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: transparent !important;
            border: none !important;
            color: #58a6ff !important;
            padding: 0px !important;
            min-height: 0px !important;
            height: auto !important;
            box-shadow: none !important;
            text-decoration: underline;
            font-size: 14px !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: transparent !important;
            color: #79c0ff !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Chord Tracker by AIrawan")

# --- HELPER FUNCTIONS ---

def apply_highpass_filter(y, sr, cutoff_freq=80):
    """High-pass filter menggunakan SOS (lebih stabil dari filtfilt standar)"""
    sos = butter(10, cutoff_freq, 'hp', fs=sr, output='sos')
    return sosfilt(sos, y)

def process_time_stretch(y, sr, rate_factor):
    """Time Stretch Profesional (Pitch Lock 100%)"""
    if rate_factor == 1.0:
        return y
    if USE_RUBBERBAND:
        return pyrb.time_stretch(y, sr, rate_factor)
    else:
        stft = librosa.stft(y)
        stft_stretched = librosa.phase_vocoder(stft, rate=rate_factor)
        return librosa.istft(stft_stretched)

def reset_tempo():
    """Callback untuk Reset Kecepatan Tempo ke 1.0x"""
    st.session_state.speed_slider = 1.0

# --- FUNGSI DETEKSI AKORD PRESISI (CENS + BEAT-SYNC) ---

def generate_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_types = {
        'maj':   {'intervals': [0, 4, 7],         'weights': [1.0, 1.0, 0.8]},
        'min':   {'intervals': [0, 3, 7],         'weights': [1.0, 1.0, 0.8]},
        'dim':   {'intervals': [0, 3, 6],         'weights': [1.0, 1.0, 0.8]},
        'aug':   {'intervals': [0, 4, 8],         'weights': [1.0, 1.0, 0.8]},
        'maj7':  {'intervals': [0, 4, 7, 11],     'weights': [1.0, 1.0, 0.8, 0.9]},
        'm7':    {'intervals': [0, 3, 7, 10],     'weights': [1.0, 1.0, 0.8, 0.9]},
        '7':     {'intervals': [0, 4, 7, 10],     'weights': [1.0, 1.0, 0.8, 0.9]},
        'dim7':  {'intervals': [0, 3, 6, 9],      'weights': [1.0, 1.0, 0.8, 0.9]},
        'm7b5':  {'intervals': [0, 3, 6, 10],     'weights': [1.0, 1.0, 0.8, 0.9]}
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
    # Ekstrak CENS (Tahan terhadap dinamika dan noise)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), bins_per_octave=36)
    
    # Sinkronisasi ke beat (Anti-jitter)
    if len(beats) > 0:
        chroma_sync = librosa.util.sync(chroma_cens, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(beats, sr=sr)
    else:
        chroma_sync = chroma_cens
        beat_times = librosa.times_like(chroma_cens, sr=sr)
        
    templates, labels = generate_chord_templates()
    similarities = np.dot(templates.T, chroma_sync)
    
    # Penalti 5% untuk akord 7ths
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
    # Reset state jika file baru diunggah
    if "uploaded_name" not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.file_id = str(time.time()) # Unique ID per unggahan
        st.session_state.speed_slider = 1.0 # Default speed
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.audio_path = tmp.name

        with st.spinner("Menganalisis BPM, Harmonic Separation & mendeteksi Akord..."):
            data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
            
            # 1. Deteksi BPM & Beats
            tempo, beats = librosa.beat.beat_track(y=data, sr=rate)
            st.session_state.detected_bpm = int(round(float(np.mean(tempo))))

            # 2. HPF (Buang Rumble) & HPSS (Buang Drum/Perkusi)
            data_hp = apply_highpass_filter(data, rate, cutoff_freq=80)
            data_harmonic, _ = librosa.effects.hpss(data_hp)
            
            # 3. Deteksi Akord via Librosa CENS + Beat-Sync
            raw_chords = detect_chords_librosa(data_harmonic, rate, beats)
            st.session_state.raw_chords = raw_chords

            # Load Audio Stereo untuk Playback & Time-Stretch
            y_full, sr_full = librosa.load(st.session_state.audio_path, sr=44100, mono=False)
            st.session_state.y_full = y_full
            st.session_state.sr_full = sr_full

    # Tampilkan Info BPM
    st.subheader("⚙️ Informasi Audio")
    st.markdown(f"**Auto BPM Original:** `{st.session_state.detected_bpm} BPM`")

    if "speed_slider" not in st.session_state:
        st.session_state.speed_slider = 1.0

    # Layout Control Bar Tempo
    col_speed, col_reset_btn, col_info = st.columns([1.8, 0.4, 1.8])
    
    with col_speed:
        speed_factor = st.slider(
            "⚡ Kecepatan Tempo", 
            min_value=0.5, 
            max_value=1.5, 
            step=0.05,
            key="speed_slider"
        )
        
    with col_reset_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        st.button("Reset", on_click=reset_tempo, help="Reset ke 1.0x", key="btn_reset_text")

    current_bpm = int(round(st.session_state.detected_bpm * speed_factor))
    with col_info:
        st.markdown(f"<p style='margin-top: 32px;'><b>Tempo Efektif:</b> <code>{current_bpm} BPM</code> ({speed_factor:.2f}x)</p>", unsafe_allow_html=True)

    # Proses Time-Stretch (Di-cache berdasarkan factor tempo)
    file_key = f"{st.session_state.file_id}_{speed_factor}"
    processed_audio_path = os.path.join(tempfile.gettempdir(), f"stretched_{file_key}.wav")
    
    if not os.path.exists(processed_audio_path):
        with st.spinner("Memproses audio & memuat Waveform..."):
            y_audio = st.session_state.y_full
            sr_audio = st.session_state.sr_full
            
            if y_audio.ndim == 2:
                left = process_time_stretch(y_audio[0], sr_audio, speed_factor)
                right = process_time_stretch(y_audio[1], sr_audio, speed_factor)
                stretched_audio = np.vstack([left, right]).T
            else:
                stretched_audio = process_time_stretch(y_audio, sr_audio, speed_factor)
                
            sf.write(processed_audio_path, stretched_audio, sr_audio)

    # Sesuaikan Timestamp Akord dengan Tempo Baru
    adjusted_chords = []
    for c in st.session_state.raw_chords:
        adjusted_chords.append({
            "time": c["time"] / speed_factor,
            "label": c["label"]
        })

    chords_json = json.dumps(adjusted_chords)

    # Encode Audio ke Base64
    with open(processed_audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # --- HTML / JS WAVESURFER PLAYER ---
    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/wavesurfer.min.js"></script>
        <script src="https://unpkg.com/wavesurfer.js@6.6.4/dist/plugin/wavesurfer.regions.min.js"></script>
        <style>
            body {{
                background-color: #0d1117;
                color: white;
                font-family: -apple-system, sans-serif;
                margin: 0;
                padding: 10px;
            }}
            .player-container {{
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 15px;
            }}
            #waveform {{
                width: 100%;
                margin-bottom: 15px;
                position: relative;
            }}
            .controls {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            button {{
                background-color: #238636;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.2s;
            }}
            button:hover {{
                background-color: #2ea043;
            }}
            button.btn-loop-active {{
                background-color: #d29922;
            }}
            button.btn-clear {{
                background-color: #21262d;
                border: 1px solid #30363d;
            }}
            button.btn-clear:hover {{
                background-color: #30363d;
            }}
            .chord-box {{
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }}
            .chord-title {{
                color: #8b949e;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .chord-val {{
                font-size: 56px;
                font-weight: bold;
                color: #58a6ff;
                margin-top: 5px;
            }}
            .wavesurfer-region {{
                border-left: 1px solid rgba(88, 166, 255, 0.5) !important;
                background-color: rgba(88, 166, 255, 0.05) !important;
            }}
            .wavesurfer-region[data-region-highlight="true"] {{
                background-color: rgba(210, 153, 34, 0.25) !important;
                border: 1px solid #d29922 !important;
            }}
            .chord-label-tag {{
                position: absolute;
                top: 2px;
                left: 3px;
                background: rgba(31, 111, 235, 0.85);
                color: #ffffff;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 4px;
                border-radius: 3px;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="player-container" id="player_{file_key}">
            <div id="waveform"></div>
            <div class="controls">
                <button onclick="wavesurfer.playPause()">Play / Pause</button>
                <button id="btnLoop" onclick="toggleLoopMode()">🔁 Loop Section: OFF</button>
                <button class="btn-clear" onclick="clearCustomLoop()">❌ Hapus Seleksi Loop</button>
            </div>
            <div class="chord-box">
                <div class="chord-title">Akord Aktif</div>
                <div id="chordDisplay" class="chord-val">-</div>
            </div>
        </div>

        <script>
            const chordData = {chords_json};
            const chordDisplay = document.getElementById('chordDisplay');
            const btnLoop = document.getElementById('btnLoop');

            let isLoopEnabled = false;
            let activeLoopRegion = null;

            const wavesurfer = WaveSurfer.create({{
                container: '#waveform',
                waveColor: '#30363d',
                progressColor: '#58a6ff',
                cursorColor: '#f0883e',
                height: 90,
                responsive: true,
                plugins: [
                    WaveSurfer.regions.create({{
                        dragSelection: true
                    }})
                ]
            }});

            wavesurfer.load('data:audio/wav;base64,{audio_b64}');

            wavesurfer.on('ready', () => {{
                const totalDuration = wavesurfer.getDuration();
                chordData.forEach((item, index) => {{
                    const nextTime = (index < chordData.length - 1) ? chordData[index + 1].time : totalDuration;
                    
                    const region = wavesurfer.addRegion({{
                        start: item.time,
                        end: nextTime,
                        drag: false,
                        resize: false
                    }});

                    if (region.element) {{
                        const tag = document.createElement('span');
                        tag.className = 'chord-label-tag';
                        tag.innerText = item.label;
                        region.element.appendChild(tag);
                    }}
                }});
            }});

            wavesurfer.on('region-created', (region) => {{
                if (region.drag || region.resize) {{
                    if (activeLoopRegion && activeLoopRegion !== region && activeLoopRegion.drag) {{
                        activeLoopRegion.remove();
                    }}
                    activeLoopRegion = region;
                    region.element.setAttribute('data-region-highlight', 'true');
                    
                    if (!isLoopEnabled) {{
                        toggleLoopMode();
                    }}
                }}
            }});

            wavesurfer.on('region-out', (region) => {{
                if (isLoopEnabled && activeLoopRegion && region === activeLoopRegion) {{
                    wavesurfer.seekTo(activeLoopRegion.start / wavesurfer.getDuration());
                    wavesurfer.play();
                }}
            }});

            wavesurfer.on('region-click', (region, e) => {{
                e.stopPropagation();
                if (!region.drag) {{
                    if (activeLoopRegion && activeLoopRegion.drag) {{
                        activeLoopRegion.remove();
                    }}
                    activeLoopRegion = region;
                    if (!isLoopEnabled) {{
                        toggleLoopMode();
                    }}
                    wavesurfer.seekTo(region.start / wavesurfer.getDuration());
                    wavesurfer.play();
                }}
            }});

            function toggleLoopMode() {{
                isLoopEnabled = !isLoopEnabled;
                if (isLoopEnabled) {{
                    btnLoop.innerText = "🔁 Loop Section: ON";
                    btnLoop.classList.add('btn-loop-active');
                }} else {{
                    btnLoop.innerText = "🔁 Loop Section: OFF";
                    btnLoop.classList.remove('btn-loop-active');
                }}
            }}

            function clearCustomLoop() {{
                if (activeLoopRegion && activeLoopRegion.drag) {{
                    activeLoopRegion.remove();
                }}
                activeLoopRegion = null;
                if (isLoopEnabled) {{
                    toggleLoopMode();
                }}
            }}

            wavesurfer.on('audioprocess', () => {{
                const currentTime = wavesurfer.getCurrentTime();
                let activeChord = "-";

                for (let i = 0; i < chordData.length; i++) {{
                    if (currentTime >= chordData[i].time) {{
                        if (i === chordData.length - 1 || currentTime < chordData[i+1].time) {{
                            activeChord = chordData[i].label;
                            break;
                        }}
                    }}
                }}
                chordDisplay.innerText = activeChord;
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(player_html, height=280)
