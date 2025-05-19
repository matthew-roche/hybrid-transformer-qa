from flask import Flask, jsonify
from flask_smorest import Api, Blueprint

from add_path import add_project_path
add_project_path()

from api.model_blueprint import model_blueprint

app = Flask(__name__)
app.config["API_TITLE"] = "Question Answering - API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_URL_PREFIX"] = "/api"
app.config["OPENAPI_REDOC_PATH"] = "/docs/redoc"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/docs" 
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.18.2/"  # Swagger UI CDN

api = Api(app)

server_blueprint = Blueprint("API", __name__, url_prefix="/api", description="api operations")

@server_blueprint.route("/health")
@server_blueprint.response(200, content_type="application/json")
def health_check():
    return jsonify({"status": "healthy"})

api.register_blueprint(server_blueprint)
api.register_blueprint(model_blueprint)

if __name__ == "__main__":
    app.run(debug=True)