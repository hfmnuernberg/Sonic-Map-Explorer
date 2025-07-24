# Sonic Map Explorer

The Sonic Map Explorer introduces an emergent interface that projects live audio signals in real time onto a two-dimensional sound map, enabling touchless and prompt-free interaction with generative AI music systems. Through a combination of spectral feature extraction, UMAP dimensionality reduction, and a learning-based mapping network, each playing gesture of the musician is visualized as a positional trajectory and simultaneously used to control a variational autoencoder that generates rhythmic patterns or melodies with near-zero latency. The system avoids predefined categories or manual parameter assignments, allowing a seamless transition from brief self-training to performative co-creation. Initial practice-based tests with improvising instrumentalists indicate that the sound map is understood both as an audio-visual feedback system and as a creative navigation surface; at the same time, they reveal optimization potential in latency and GUI ergonomics. Planned next development steps include an open-source release as a Max for Live device, an automatic “novelty check,” and GRU-based mapping models. This paper positions the Sonic Map Explorer as a reference model for emergent human-AI interaction in live music.

## Usage
- Use the M4L-Device from the [releases](https://github.com/hfmnuernberg/Sonic-Map-Explorer/releases/tag/M4L) for direct usage inside Ableton Live.
- Or open the Max Project for Code insights.
  - Dependencies for Max Project:
    - Flucoma Package
    - Node.js Packages: tensorflow/tfjs-node, tonejs/midi  
