#!/usr/bin/env python3

"""ttsdemo.py

There's either the "gtts" module (haven't tried it) to be used,
or speech synthesis frameworks that can be called (subprocess).

    "espeak {message}" # speaks immediately
    "spd-say {message}" # speaks immediately

I usually use the festival speech synthesizer:
    https://www.cstr.ed.ac.uk/projects/festival/index.html

    "text2wave -o {outputFileName}.wav" # writes to wave file (message through pipe)
    "echo '(SayText "{message}")' | festival --pipe" # speaks immediately

If you have some working python-only tts functions,
feel free to include them in this script.

Finn M Glas, 2021-09-12 02:45:00 CEST
"""

# main demo

if __name__ == "__main__":
    print("Not implemented.")
