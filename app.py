with st.spinner("Menganalisis akord & mendeteksi BPM tempo..."):
            data, rate = librosa.load(st.session_state.audio_path, sr=44100, mono=True)
            
            # 1. Deteksi BPM
            tempo, _ = librosa.beat.beat_track(y=data, sr=rate)
            st.session_state.detected_bpm = int(round(float(tempo)))

            # 2. Deteksi Akord via Chroma Librosa (Cloud-Friendly & Tanpa Compiler C++)
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
