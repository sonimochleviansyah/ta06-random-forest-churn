# TA-07: Q-Learning untuk Kurir Paket Cerdas (Grid 4x4)

Soni Moch Leviansyah (NIM 301240030) - Pembelajaran Mesin, Modul 07 Reinforcement Learning.

Environment kustom Python murni (tanpa Gymnasium): kurir mengambil paket di P=(0,3), mengantar ke D=(3,0), menghindari zona banjir (1,1) dan (2,2).
32 state (16 posisi x 2 status paket), 6 aksi, Q-Table 32x6.

## Isi
- `kurir_env.py` - environment (`step_from(state, action)` -> `new_state, reward, done`)
- `qlearning.py` - Q-Learning, epsilon-greedy dengan peluruhan eksponensial, moving average, rollout greedy
- `run_all.py` - pelatihan dasar, kurva pembelajaran, visualisasi policy, eksperimen gamma
- `exp_gamma.py` - eksperimen hyperparameter gamma 0.9 vs 0.1 (10 seed)
- `TA-07_Q_Learning_Kurir.ipynb` - notebook (Jupyter/Colab) langkah 1-5
- `gambar/` - hasil grafik

## Menjalankan
```
pip install -r requirements.txt
python run_all.py
python exp_gamma.py
```

## Hasil utama (alpha=0.8, gamma=0.9, 2000 episode, seed 42)
- Konvergensi (MA100 >= 95% plateau): episode 777
- Policy greedy: 11 langkah, reward 31 (maksimum)
- Eksperimen gamma (10 seed): keduanya berhasil 10/10; selisih Q terbaik-kedua di state awal 4.17 (gamma 0.9) vs 0.011 (gamma 0.1)
