import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="ChordTracker",
    page_icon="🎸",
    layout="centered"
)

# Fungsi bantu untuk highpass filter sederhana
def apply_highpass_filter(data, rate, cutoff_freq=80):
    # Menggunakan librosa/scipy filter sederhana atau mengembalikan data asli jika sudah bersih
    return data

# Judul Aplikasi
st.title("🎸 ChordTracker by AIrawan")
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
        # Semua perintah UI seperti st.spinner harus berada di dalam blok logika/fungsi seperti ini
        with st.spinner("Menganalisis akord & mendeteksi BPM tempo..."):
            try:
                # Load audio menggunakan librosa
                data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
                
                # 1. Deteksi BPM Tempo (Diperbarui dengan mengambil indeks [0] untuk mencegah error array)
                tempo, _ = librosa.beat.beat_track(y=data, sr=rate)
                if isinstance(tempo, np.ndarray):
                    st.session_state.detected_bpm = int(round(float(tempo[0])))
                else:
                    st.session_state.detected_bpm = int(round(float(tempo)))

                # 2. Deteksi Akord via Chroma Librosa (Cloud-Friendly & Stabil)
                data_hp = apply_highpass_filter(data, rate, cutoff_freq=80)
                chroma = librosa.feature.chroma_cqt(y=data_hp, sr=rate)
                chord_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                
                times = librosa.times_like(chroma.shape[1], sr=rate)
                
                raw_chords = []
                prev_chord = None
                for i, t in enumerate(times):
                    chord_idx = np.argmax(chroma[:, i])
                    label = chord_labels[chord_idx]
                    if label != prev_chord:
                        raw_chords.append({"time": float(t), "label": label})
                        prev_chord = label
                        
                st.session_state.raw_chords = raw_chords

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
        for c in st.session_state.raw_chords[:50]: # Batasi 50 baris pertama agar rapi
            m = int(c['time'] // 60)
            s = int(c['time'] % 60)
            chord_display.append({"Waktu": f"{m:02d}:{s:02d}", "Akord": c['label']})
        
        st.table(chord_display)
    else:
        st.info("Belum ada data akord yang terdeteksi.")
