from flask import Flask, send_from_directory

app = Flask(__name__)


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

if __name__ == "__main__":
    app.run()