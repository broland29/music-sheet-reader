from music21 import note, stream
import os

INPUT_PATH = "/home/broland/Documents/ut/ip/music_sheet_reader/cmake-build-debug/notes.txt"
OUTPUT_PATH = "/home/broland/Documents/ut/ip/music_sheet_reader/cmake-build-debug/notes.midi"


def map_duration(dur_char):
    match dur_char:
        case 'W':
            return 4
        case 'H':
            return 2
        case 'Q':
            return 1
        case 'E':
            return .5
        case 'S':
            return .25
        case _:
            print(f"Unexpected duration {dur_char} in map_duration, returning 1")
            return 1


if __name__ == '__main__':
    try:
        file = open(INPUT_PATH)
    except FileNotFoundError:
        print(f"File {INPUT_PATH} not found!")
        exit(1)

    lines = file.readlines()
    print(lines)
    notes = []
    for line in lines:
        note_and_octave = line[:2]
        duration = map_duration(line[2])
        notes.append(note.Note(note_and_octave, quarterLength=duration))

    s = stream.Stream()
    s.append(notes)
    s.write('midi', fp=OUTPUT_PATH)

    os.system(f"vlc {OUTPUT_PATH}")
