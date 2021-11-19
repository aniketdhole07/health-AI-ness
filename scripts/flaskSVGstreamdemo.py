#!/usr/bin/env python3

"""flaskSVGstreamdemo.py

This demonstrates SVG- and data streaming between a dummy Web/JS frontend
and a python3 flask backend, both intermingled within this script.

Resulting http server (base http://127.0.0.1:5000):

    Index           /
    Random          /api/random
    Random Line     /api/randomsvg

By changing the random svg stream to a rendered stream of coordinates from the
pose-estimator, moving the static HTML / CSS / JS code to external files, and
improving the UI appearance, this demo could act as a core part of the prototype.

Finn M Glas, 2021-09-21 00:09:00 CEST
"""

## Static HTML / CSS / JS Code

STATIC_CSS_STYLE = """
* {
  font-family: sans-serif;
  color: #555;
  vertical-align:top;
}
body {
  max-width: 600px;
  margin: auto;
  padding: 10px;
  margin-top: 50px;
  margin-bottom: 50px;
}
h1 { font-size: 24px; margin: 0; margin-bottom: 10px; letter-spacing: 1.5px; }
h2 { font-size: 18px; margin: 0; margin-bottom: 5px; font-weight: bolder; }
"""

STATIC_HTML_TITLE = "HealthAIness Stream Demo using Flask"

STATIC_HTML_HEAD = f"""
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta charset="utf-8">
  <link rel="shortcut icon" href="logo.png">
  <title>{STATIC_HTML_TITLE}</title>
  <style>{STATIC_CSS_STYLE}</style>

  <!-- Import Font awesome icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/css/all.min.css">
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/js/all.min.js" charset="utf-8"></script>
</head>"""

STATIC_JS_RELOADING = """
function httpGet(theUrl) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", theUrl, false ); // false for synchronous request
    xmlHttp.send( null );
    return xmlHttp.responseText;
}

var shouldFetch = false; // State of the demo
var frameRate = 30

document.getElementById('rnum').innerHTML = "0";
document.getElementById('rsvg').innerHTML = httpGet("/api/randomsvg");

function updateFrame() {
    if (shouldFetch) {
        document.getElementById('rnum').innerHTML = httpGet("/api/random");
        document.getElementById('rsvg').innerHTML = httpGet("/api/randomsvg");
    }
    setTimeout(updateFrame, 1000/frameRate);
}

updateFrame()
"""

STATIC_HTML_BODY = f"""<body>
    <h1>HealthAINess Stream Demo using Flask</h1>

    A random number (for refresh-rate tests): <span id="rnum"></span>

    <br>

    A random SVG image:

    <div id="rsvg"></div>

    <script type="text/javascript">{STATIC_JS_RELOADING}</script>

    <button onClick="shouldFetch=!shouldFetch;">Toggle Fetching</button>
    <input type="number" min="1" max="100" value="30" title="Frame rate" onkeyup="frameRate = this.value" oninput="frameRate = this.value"></input>
</body>"""

STATIC_HTML = f"""<html>{STATIC_HTML_HEAD}{STATIC_HTML_BODY}</html>"""

## The actual backend

from flask import Flask
import random

app = Flask(__name__)


@app.route("/")
def page_index():
    """The main / index page."""
    return STATIC_HTML


@app.route("/api/random")
def page_random():
    """Generate a random number for demo script (not needed in prototype)"""
    return str(random.randint(0, 255))


@app.route("/api/randomsvg")
def page_randomSVG():
    """Randomly scatter a line somewhere in a SVG.
    This would have to be replaced with a template-string svg,
    representing a human body.
    """

    height = 210
    width = 500
    x1 = random.randint(0, 200)
    y1 = random.randint(0, 200)
    x2 = random.randint(0, 200)
    y2 = random.randint(0, 200)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return f"""<svg height=\"{height}" width=\"{width}"">
  <line x1=\"{x1}" y1=\"{y1}" x2=\"{x2}" y2=\"{y2}" style="stroke:rgb({r},{g},{b});stroke-width:2" />
</svg>"""


if __name__ == "__main__":
    app.run()
