import json
import os
import datetime as dt

from flask import Flask, Response, render_template, request, session, url_for, jsonify, redirect
from flask_cors import CORS
import db
from camera import VideoCamera
from pose_model.demo import get_frames
usedCameraStream = None # stores wrapper of either a webcam or luxonis board

# state variables for the exercise page
enabledVideoStream = True
enabledVoiceCommand = False

def create_app(test_config=None):
    """Entrypoint of execution."""
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY="development",
        DATABASE=os.path.join(app.instance_path, "healthainess.sqlite"),
    )

    # if test_config is None:
    #     app.config.from_pyfile("config.py", silent=True)
    # else:
    #     app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)

    ## --- Template pages --- ##

    @app.route("/")
    def page_start():
        return render_template("page_start.html")

    @app.route("/exercise")
    def page_exercise():
        return render_template("page_exercise.html")

    @app.route("/report")
    def page_report():
        return render_template("page_report.html")

    @app.route("/config")
    def page_config():
        return render_template("page_config.html")

    ## --- UI state synchronisation --- ##

    @app.route("/exercise/var/enabledVideoStream")
    def var_exercise_stream():
        return str(int(enabledVideoStream))

    @app.route("/exercise/var/enabledVoiceCommand")
    def var_exercise_voice():
        return str(int(enabledVoiceCommand))

    @app.route("/exercise/var/enabledVideoStream/set", methods=['POST'])
    def set_var_exercise_stream():
        """Receives input from the play/ pause button in UI."""

        global usedCameraStream
        global enabledVideoStream

        if usedCameraStream == None:
            usedCameraStream = VideoCamera()

        enabledVideoStream = request.get_json()['state']

        return jsonify(state=enabledVideoStream)

    @app.route("/exercise/var/enabledVoiceCommand/set", methods=['POST'])
    def set_var_exercise_voice():
        """Receives input from the start/ stop voice command button in UI."""

        global enabledVoiceCommand

        enabledVoiceCommand = request.get_json()['state']
        
        return jsonify(state=enabledVoiceCommand)

    ## --- Camera / data stream endpoints --- ##

    def video_get():
        """Get a frame and encode it."""

        global usedCameraStream
        global enabledVideoStream
        if usedCameraStream == None:
            usedCameraStream = VideoCamera()

        while True:
            if (enabledVideoStream):
                frame = usedCameraStream.get_frame()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    @app.route('/video_stream')
    def video_stream():
        return Response(get_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    ## --- API endpoints --- #

    @app.route('/api/exercises_done/fetchAll', methods=['GET'])
    def exercises_done_fetchAll():
        _db = db.get_db()
        exercises_done = _db.execute(
            'SELECT id, name, duration_s, quality, finished_at'
            ' FROM exercises_done'
            ' ORDER BY finished_at DESC'
        ).fetchall()

        result = [dict(r) for r in exercises_done]
        return Response(json.dumps(result, indent=4, sort_keys=False, default=str), status=200,
                        mimetype='application/json')

    #
    # http://0.0.0.0:5000//api/exercises_done/fetchFromTo?from=2020-02-01 12:00:00&to=2021-02-01 12:00:00
    #
    # {
    #     "from": "2020-01-01 00:00:00",
    #     "to": "2021-01-01 00:00:00"
    # }
    @app.route('/api/exercises_done/fetchFromTo', methods=['GET', 'POST'])
    def exercises_done_fetchFromTo():
        if request.is_json:
            content = request.get_json()
            fetch_from = content['from']
            fetch_to = content['to']
        else:
            fetch_from = request.args.get('from')
            fetch_to = request.args.get('to')

        if not fetch_from:
            yesterday = dt.date.today() - dt.timedelta(days=7)
            fetch_from = yesterday

        if not fetch_to:
            fetch_to = dt.datetime.now()

        _db = db.get_db()
        exercises_done = _db.execute(
            'SELECT id, name, duration_s, quality, finished_at'
            ' FROM exercises_done'
            ' where '
            ' finished_at between ? and ?'
            ' ORDER BY finished_at ASC',
            (fetch_from, fetch_to)
        ).fetchall()

        result = [dict(r) for r in exercises_done]
        return Response(json.dumps(result, indent=4, sort_keys=False, default=str), status=200,
                        mimetype='application/json')

    @app.route('/api/exercises_done/create', methods=['POST'])
    def exercises_done_create():
        error = None

        json_data = request.get_json(silent=True)
        if json_data is None:
            error = True
        else:
            name = json_data["name"]
            duration_s = json_data["duration_s"]
            quality = json_data["quality"]
            finished_at = json_data["finished_at"]

            if name is None:
                error = True
            if duration_s is None:
                error = True
            if quality is None:
                error = True
            if finished_at is None:
                error = True

            if error is None:
                _db = db.get_db()
                _db.execute(
                    'INSERT INTO exercises_done (name, duration_s, quality, finished_at)'
                    ' VALUES (?, ?, ?, ?)',
                    (name, duration_s, quality, finished_at)
                )
                _db.commit()
                return Response("{'status':'OK'}", status=200, mimetype='application/json')

        return Response("{'status':'Error'}", status=400, mimetype='application/json')

    @app.route('/api/exercises_done/deleteAll', methods=['GET', 'POST'])
    def exercises_done_deleteAll():
        _db = db.get_db()
        _db.execute('DELETE FROM exercises_done')
        _db.commit()
        return Response("{'status':'OK'}", status=200, mimetype='application/json')

    # http://0.0.0.0:5000/api/report/exercisesDoneByDayFromTo?from=2020-02-01 12:00:00&to=2021-02-01 12:00:00
    #
    # {
    #     "from": "2020-01-01 00:00:00",
    #     "to": "2021-01-01 00:00:00"
    # }
    @app.route('/api/report/exercisesDoneByDayFromTo', methods=['GET', 'POST'])
    def report_exercises_done_by_day_from_to():
        if request.is_json:
            content = request.get_json()
            fetch_from = content['from']
            fetch_to = content['to']
        else:
            fetch_from = request.args.get('from')
            fetch_to = request.args.get('to')

        if not fetch_from:
            yesterday = dt.date.today() - dt.timedelta(days=7)
            fetch_from = yesterday

        if not fetch_to:
            fetch_to = dt.datetime.now()

        try:
            _db = db.get_db()
            exercises_done = _db.execute(
                """
                SELECT
                    -- id,
                    name,
                    -- duration_s,
                    -- quality,
                    -- finished_at,
                    strftime( '%d', finished_at) AS d,
                    strftime( '%m,', finished_at) AS m,
                    strftime( '%Y', finished_at) AS y,
                    strftime( '%Y%m%d', finished_at) AS ymd,
                    strftime( '%Y-%m-%d', finished_at) AS ymdLabel,
                    SUM(duration_s) AS duration_s_acc,
                    AVG(quality) AS quality_acc,
                    COUNT(*) AS cnt
                FROM exercises_done
                WHERE finished_at between ? and ?
                GROUP BY name, ymd
                ORDER BY ymd ASC
                """,
                (fetch_from, fetch_to)
            ).fetchall()

            result = [dict(r) for r in exercises_done]
            return Response(json.dumps(result, indent=4, sort_keys=False, default=str), status=200,
                            mimetype='application/json')
        except Exception as e:
            #print(type(e))
            print(e)

        return Response("{'status':'Error'}", status=400, mimetype='application/json')

        return app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', threaded=True,debug=True)