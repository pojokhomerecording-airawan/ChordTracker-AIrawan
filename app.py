import streamlit as st
import librosa
import numpy as np
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="ChordTracker",
    page_icon="🎸",
    layout="centered"
)

# Judul Aplikasi
st.title("🎸 ChordTracker & Audio Analyzer")
st.write("Aplikasi deteksi akord, tempo BPM, dan pemutar audio.")

# Inisialisasi session state jika belum ada
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "raw_chords" not in st.session_state:
    st.session_state.raw_chords = []
if "detected_bpm" not in st.session_state:
    st.session_state.detected_bpm = 0
if "y_full" not in st.session_state:
    st.session_state.y_full = None
if "sr_full" not in st.session_state:
    st.session_state.sr_full = None

# Bagian Upload File Audio
uploaded_file = st.file_uploader("Upload file audio (MP3 / WAV)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Simpan file sementara ke disk
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.audio_path = temp_path
    st.success("File audio berhasil di-upload!")

# Tombol Proses Analisis
if st.session_state.audio_path is not None:
    if st.button("Mulai Analisis Audio"):
        with st.spinner("Menganalisis akord & mendeteksi BPM tempo..."):
            try:
                # Load audio menggunakan librosa
                data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
                
                # 1. Deteksi BPM Tempo (Mengatasi error array to scalar)
                tempo, _ = librosa.beat.beat_track(y=data, sr=rate)
                if isinstance(tempo, np.ndarray):
                    st.session_state.detected_bpm = int(round(float(tempo[0])))
                else:
                    st.session_state.detected_bpm = int(round(float(tempo)))

                # 2. Deteksi Akord Lanjutan via Template Matching (Mayor & Minor)
                # Memisahkan elemen harmonik dari audio (drum & perkusi diabaikan)
                y_harmonic, _ = librosa.effects.hpss(data)
                
                # Ekstraksi fitur chroma HANYA dari instrumen harmonik
                chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=rate)
                
                # Definisi label dan template akord (Mayor dan Minor)
                chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                chord_templates = []
                chord_labels = []
                
                for i in range(12):
                    # Pola Akord Mayor (Root, Major 3rd, Perfect 5th)
                    t_maj = np.zeros(12)
                    t_maj[[i, (i+4)%12, (i+7)%12]] = 1
                    chord_templates.append(t_maj)
                    chord_labels.append(chroma_notes[i])
                    
                    # Pola Akord Minor (Root, Minor 3rd, Perfect 5th)
                    t_min = np.zeros(12)
                    t_min[[i, (i+3)%12, (i+7)%12]] = 1
                    chord_templates.append(t_min)
                    chord_labels.append(f"{chroma_notes[i]}m")
                
                # Ubah ke matriks agar bisa dihitung matematis
                chord_templates = np.array(chord_templates).T 
                
                # Hitung skor kecocokan antara audio dengan pola akord Mayor/Minor
                chord_scores = np.dot(chord_templates.T, chroma)
                times = librosa.times_like(chroma.shape[1], sr=rate)
                
                raw_chords = []
                prev_chord = None
                
                for i, t in enumerate(times):
                    # Ambil akord dengan skor tertinggi di waktu tersebut
                    best_chord_idx = np.argmax(chord_scores[:, i])
                    label = chord_labels[best_chord_idx]
                    
                    if label != prev_chord:
                        raw_chords.append({"time": float(t), "label": label})
                        prev_chord = label
                        
                # Filter: Gabungkan akord yang berkedip terlalu cepat (transisi/noise)
                filtered_chords = []
                min_duration = 0.5 # Akord harus bertahan minimal setengah detik
                for i in range(len(raw_chords)):
                    current_time = raw_chords[i]['time']
                    next_time = raw_chords[i+1]['time'] if i+1 < len(raw_chords) else times[-1]
                    
                    if (next_time - current_time) >= min_duration:
                        filtered_chords.append(raw_chords[i])

                st.session_state.raw_chords = filtered_chords

                # Load Audio Stereo untuk Playback
                y_full, sr_full = librosa.load(st.session_state.audio_path, sr=44100, mono=False)
                st.session_state.y_full = y_full
                st.session_state.sr_full = sr_full
                
                st.success("Analisis selesai!")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses audio: {e}")

# Tampilkan Hasil Jika Sudah Dianalisis
if st.session_state.detected_bpm > 0:
    st.markdown("---")
    st.subheader("📊 Hasil Analisis")
    st.metric(label="Perkiraan Tempo (BPM)", value=st.session_state.detected_bpm)
    
    st.write("### Daftar Perubahan Akord:")
    if st.session_state.raw_chords:
        # Tampilkan dalam bentuk tabel atau list ringkas
        chord_display = []
        for c in st.session_state.raw_chords[:150]: # Batasi jumlah baris agar tidak lag
            m = int(c['time'] // 60)
            s = int(c['time'] % 60)
            chord_display.append({"Waktu": f"{m:02d}:{s:02d}", "Akord": c['label']})
        
        st.table(chord_display)
    else:
        st.info("Belum ada data akord yang terdeteksi.")
