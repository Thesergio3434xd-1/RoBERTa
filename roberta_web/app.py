from flask import Flask

from .config import Config
from .controllers.main import main_bp


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    app.register_blueprint(main_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)