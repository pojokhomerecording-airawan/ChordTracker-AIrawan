# 2. Deteksi Akord Lanjutan via Template Matching (Mayor & Minor)
                # Memisahkan elemen harmonik dari audio agar suara drum/bass pukul tidak mengganggu akord
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
                        
                # Filter: Gabungkan akord yang berkedip terlalu cepat (transisi/noise gitar)
                filtered_chords = []
                min_duration = 0.5 # Akord harus bertahan minimal setengah detik
                for i in range(len(raw_chords)):
                    current_time = raw_chords[i]['time']
                    next_time = raw_chords[i+1]['time'] if i+1 < len(raw_chords) else times[-1]
                    
                    if (next_time - current_time) >= min_duration:
                        filtered_chords.append(raw_chords[i])

                st.session_state.raw_chords = filtered_chords
