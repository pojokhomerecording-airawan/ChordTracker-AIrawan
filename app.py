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
st.write("Aplikasi deteksi akord (termasuk ekstensi 7th), tempo BPM, dan pemutar audio.")

# Inisialisasi session state jika belum ada
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "raw_chords" not in st.session_state:
    st.session_state.raw_chords = []
if "detected_bpm" not in st.session_state:
    st.session_state.detected_bpm = 0

# Bagian Upload File Audio
uploaded_file = st.file_uploader("Upload file audio (MP3 / WAV)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
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
                
                # 1. Deteksi BPM Tempo
                tempo, _ = librosa.beat.beat_track(y=data, sr=rate)
                if isinstance(tempo, np.ndarray):
                    st.session_state.detected_bpm = int(round(float(tempo[0])))
                else:
                    st.session_state.detected_bpm = int(round(float(tempo)))

                # 2. Deteksi Akord Lanjutan via Template Matching (+ Ekstensi 7th)
                y_harmonic, _ = librosa.effects.hpss(data)
                chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=rate)
                
                chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                chord_templates = []
                chord_labels = []
                
                for i in range(12):
                    # 1. Major (Triad) [0, 4, 7]
                    t_maj = np.zeros(12)
                    t_maj[[i, (i+4)%12, (i+7)%12]] = 1
                    chord_templates.append(t_maj)
                    chord_labels.append(chroma_notes[i])
                    
                    # 2. Minor (Triad) [0, 3, 7]
                    t_min = np.zeros(12)
                    t_min[[i, (i+3)%12, (i+7)%12]] = 1
                    chord_templates.append(t_min)
                    chord_labels.append(f"{chroma_notes[i]}m")
                    
                    # 3. Dominant 7th [0, 4, 7, 10]
                    t_dom7 = np.zeros(12)
                    t_dom7[[i, (i+4)%12, (i+7)%12, (i+10)%12]] = 1
                    chord_templates.append(t_dom7)
                    chord_labels.append(f"{chroma_notes[i]}7")
                    
                    # 4. Major 7th [0, 4, 7, 11]
                    t_maj7 = np.zeros(12)
                    t_maj7[[i, (i+4)%12, (i+7)%12, (i+11)%12]] = 1
                    chord_templates.append(t_maj7)
                    chord_labels.append(f"{chroma_notes[i]}maj7")
                    
                    # 5. Minor 7th [0, 3, 7, 10]
                    t_m7 = np.zeros(12)
                    t_m7[[i, (i+3)%12, (i+7)%12, (i+10)%12]] = 1
                    chord_templates.append(t_m7)
                    chord_labels.append(f"{chroma_notes[i]}m7")

                # Normalisasi matriks template & chroma (Cosine Similarity)
                chord_templates = np.array(chord_templates)
                norms = np.linalg.norm(chord_templates, axis=1, keepdims=True)
                chord_templates_norm = chord_templates / norms
                
                chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-6)
                
                # Hitung skoring kecocokan
                chord_scores = np.dot(chord_templates_norm, chroma_norm)
                times = librosa.times_like(chroma.shape[1], sr=rate)
                
                raw_chords = []
                prev_chord = None
                
                for i, t in enumerate(times):
                    best_chord_idx = np.argmax(chord_scores[:, i])
                    label = chord_labels[best_chord_idx]
                    
                    if label != prev_chord:
                        raw_chords.append({"time": float(t), "label": label})
                        prev_chord = label
                        
                # Filter noise/transisi cepat (durasi minimal 0.6 detik)
                filtered_chords = []
                min_duration = 0.6 
                for i in range(len(raw_chords)):
                    current_time = raw_chords[i]['time']
                    next_time = raw_chords[i+1]['time'] if i+1 < len(raw_chords) else times[-1]
                    
                    if (next_time - current_time) >= min_duration:
                        filtered_chords.append(raw_chords[i])

                st.session_state.raw_chords = filtered_chords
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
        chord_display = []
        for c in st.session_state.raw_chords[:150]:
            m = int(c['time'] // 60)
            s = int(c['time'] % 60)
            chord_display.append({"Waktu": f"{m:02d}:{s:02d}", "Akord": c['label']})
        
        st.table(chord_display)
    else:
        st.info("Belum ada data akord yang terdeteksi.")
