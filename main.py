
# main.py
import os
try:
    from app import create_app

    app = create_app()

except Exception as e:
    print("✗ Fallback activated due to error:")
    import traceback

    traceback.print_exc()

    from flask import Flask

    app = Flask(__name__)


    @app.route("/")
    def hello():
        return "Hello World! (fallback mode)"


    @app.route("/healthz")
    def healthz():
        return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
