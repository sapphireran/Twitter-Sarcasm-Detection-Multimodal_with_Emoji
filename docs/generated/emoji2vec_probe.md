# Emoji2vec probe

## emoji2vec_twitter.bin
header: 1661 vectors × 200 dim, bytes=1339722
loaded: 1661 × 200
  😂 → 😹 0.688, 😢 0.600, 😭 0.511, 😊 0.461, 📚 0.446
  😒 → 😞 0.508, 🙎 0.485, 🙁 0.482, 😟 0.481, 😨 0.470
  😭 → 😿 0.655, 😢 0.616, 😞 0.524, 😂 0.511, 😦 0.497
  ❤: OOV
  🔫 → ⚗ 0.447, 🎉 0.415, 🤗 0.408, 🏧 0.408, 💥 0.407

## emoji2vec.bin
header: 1661 vectors × 300 dim, bytes=2004122
loaded: 1661 × 300
  😂 → 😹 0.872, 😃 0.693, 😆 0.683, 😭 0.671, 😿 0.653
  😒 → 😞 0.646, 😠 0.596, 😬 0.561, 🙎 0.561, 🙍 0.555
  😭 → 😿 0.836, 😢 0.689, 😹 0.677, 😂 0.671, 😦 0.622
  ❤: OOV
  🔫 → 🗡 0.603, 🔪 0.519, 🏹 0.365, 🚄 0.355, 🚅 0.336

## mean-pool demo (200d twitter table)
tweet: I loovee when people text back ... 😒 #sarcastictweet
tokens: ['i', 'loovee', 'when', 'people', 'text', 'back', '...', '😒', '#sarcastictweet']
in-vocab emoji: ['😒']
mean L2: 2.8968
nearest to mean: 😒 1.000, 😞 0.508, 🙎 0.485, 🙁 0.482, 😟 0.481
