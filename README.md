## Music Sheet Reader
An OpenCV project for transforming an image into a MIDI file.
  - Technical University of Cluj-Napoca, Image Processing, Final Project
  - Professor Varga Robert
  - OpenCV C++ library, performing manually implemented transformations such as:
    - Horizontal Projection
    - Opening (Erosion + Dilation)
    - Connected Component Labeling (BFS)
   - C++ program outputs a text file notes.txt
   - Python script parses notes.txt and uses music21 library to generate MIDI, then plays it using VLC
   - Limitations:
	   - only quarter and eighth notes with beams are recognized
	   - uses some hardcoded values which are highly input-specific

5/26/2023
