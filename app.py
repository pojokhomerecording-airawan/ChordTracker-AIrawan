import streamlit as st
import numpy as np
import librosa
import scipy.signal
import tempfile
import os

# Set Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="ChordTracker - Librosa Engine",
    page_icon="🎸",
    layout="wide"
)

# --- FUNGSI PEMBERSIH AUDIO & SINKRONISASI ---
def apply_highpass_filter(y, sr, cutoff_freq=80):
    """Membersihkan rumbling/noise frekuensi sangat rendah (sub-bass)."""
    sos = scipy.signal.butter(10, cutoff_freq, 'hp', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, y)

# --- FUNGSI DETEKSI AKORD DENGAN LIBROSA (DIOPTIMALKAN) ---
def generate_chord_templates():
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Memberikan bobot spesifik. Nada ke-5 sedikit dikurangi karena secara alami 
    # selalu muncul sebagai overtones (harmonik) dari nada dasar (Root).
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
            
            # Normalisasi
            norm = np.linalg.norm(template)
            if norm > 0:
                template = template / norm
            
            # Format nama akord
            if chord_name == 'maj': label = root
            elif chord_name == 'min': label = f"{root}m"
            else: label = f"{root}{chord_name}"
                
            templates.append(template)
            labels.append(label)
            
    return np.array(templates).T, labels

def detect_chords_librosa(y, sr, beats):
    # 1. Ekstrak Chroma CENS (Sangat optimal untuk chord recognition dibanding CQT/STFT)
    # Turunkan fmin ke C2 agar bass/root note tertangkap lebih jelas
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), bins_per_octave=36)
    
    # 2. Sinkronisasi Chroma ke Ketukan (Beat-Synchronous)
    # Ini menghilangkan deteksi frame-by-frame yang bergetar (jitter)
    if len(beats) > 0:
        chroma_sync = librosa.util.sync(chroma_cens, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(beats, sr=sr)
    else:
        chroma_sync = chroma_cens
        beat_times = librosa.times_like(chroma_cens, sr=sr)
    
    # 3. Hitung Similaritas (Template Matching)
    templates, labels = generate_chord_templates()
    similarities = np.dot(templates.T, chroma_sync)
    
    # 4. Beri penalti 5% pada akord 4-nada (7ths)
    # Mencegah overtones dari triad terbaca sebagai 7th secara tidak sengaja
    for idx, label in enumerate(labels):
        if any(x in label for x in ['7', 'maj7', 'm7', 'dim7', 'm7b5']):
            similarities[idx, :] *= 0.95
    
    # 5. Ambil akord dengan kecocokan tertinggi per ketukan (beat)
    best_matches = np.argmax(similarities, axis=0)
    
    # 6. Konversi ke array dictionary dengan timestamp berdasarkan waktu ketukan
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

# --- CALLBACK UNTUK RESET STATE ---
def reset_analysis():
    for key in ['raw_chords', 'detected_bpm', 'audio_path', 'y_full', 'sr_full']:
        if key in st.session_state:
            del st.session_state[key]

# --- INISIALISASI SESSION STATE ---
if 'raw_chords' not in st.session_state:
    st.session_state.raw_chords = None
if 'detected_bpm' not in st.session_state:
    st.session_state.detected_bpm = None

# --- TAMPILAN UTAMA ---
st.title("🎸 ChordTracker")
st.caption("Deteksi Akord & BPM Presisi berbasis Librosa (Beat-Synchronous CENS)")

uploaded_file = st.file_uploader("Unggah File Audio (MP3 / WAV)", type=["mp3", "wav"], on_change=reset_analysis)

if uploaded_file is not None:
    # Simpan file sementara
    if 'audio_path' not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.audio_path = tmp_file.name

    # Proses Analisis
    if st.session_state.raw_chords is None:
        with st.spinner("Menganalisis ketukan & mendeteksi akord presisi..."):
            # Load audio mono untuk analisis
            data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
            
            # 1. Deteksi BPM & Beats (Kompatibel dengan Librosa 0.10+)
            tempo, beats = librosa.beat.beat_track(y=data, sr=rate)
            st.session_state.detected_bpm = int(round(float(np.mean(tempo))))

            # 2. Filter High-Pass dan HPSS (Pisahkan instrumen dari perkusi/drum)
            data_hp = apply_highpass_filter(data, rate, cutoff_freq=80)
            data_harmonic, _ = librosa.effects.hpss(data_hp)
            
            # 3. Deteksi Akord via Librosa menggunakan Beat-Synchronous
            raw_chords = detect_chords_librosa(data_harmonic, rate, beats)
            st.session_state.raw_chords = raw_chords

            # Load Audio Stereo untuk Playback
            y_full, sr_full = librosa.load(st.session_state.audio_path, sr=44100, mono=False)
            st.session_state.y_full = y_full
            st.session_state.sr_full = sr_full

    # Pemutaran Audio
    st.audio(uploaded_file, format="audio/mp3")

    # Tampilkan Informasi Ringkasan
    st.subheader("📌 Hasil Analisis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tempo Terdeteksi (BPM)", st.session_state.detected_bpm)
    with col2:
        st.metric("Jumlah Perubahan Akord", len(st.session_state.raw_chords))

    # Tampilkan Tabel Perjalanan Akord
    st.subheader("🎼 Timeline Akord")
    
    # Format data ke tabel yang bersih
    chord_data_display = [
        {
            "Waktu (detik)": f"{item['time']:.2f} s",
            "Menit:Detik": f"{int(item['time'] // 60):02d}:{int(item['time'] % 60):02d}",
            "Akord": item['label']
        }
        for item in st.session_state.raw_chords
    ]
    
    st.dataframe(chord_data_display, use_container_width=True)
