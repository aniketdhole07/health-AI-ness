#!/usr/bin/env python3

"""speechcommandsdemo.py

A base script for the speech command (speech recognition) feature of out prototype.
To run this you need the speech_recognition module installed:

  pip3 install SpeechRecognition

This can be run in a loop-thread feeding a queue.Queue for voice commands.
Minor modifications can be made to use the bing / microsoft sr engine.

Finn M Glas, 2021-09-11 21:04:00
"""

# reuseable code

import speech_recognition as sr


def getSpeechCommand() -> dict:
    """Records and recognizes speech using the google sr api.

    :returns: A dict of likely speech recognition matches and confidences.
    """

    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        results = r.recognize_google(audio, show_all=True)
        # show_all=False -> likeliest result is chosen
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

    return results


# main demo

if __name__ == "__main__":
    print("Say something into your microphone :)")

    command = getSpeechCommand()

    if command:
        print("Closest match:", command["alternative"][0]["transcript"])
        print("Confidence:", command["alternative"][0]["confidence"])
    else:
        print("Nothing recognized, try again.")
