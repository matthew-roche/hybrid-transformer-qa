"""
This file is referenced from scripts/dev_api.py

Serves the models
"""
from flask import jsonify
from flask_smorest import Blueprint
import torch, sys, pathlib, time

from scripts.add_path import add_project_path
add_project_path()

# import model inference methods
from api.model_init import torch_device, load_qa_model, load_classifier_to_device, classifier_inference, hybrid_inference
# import model schema to validate input, and structure output for swagger
from api.model_schema import ModelInputSchema, ClassifierOutputSchema, ModelOutputSchema

# run once
start_time = time.time()
print(f"Loading models to {torch_device()}")
qa_model, qa_tokenizer = load_qa_model()
lap_time = time.time()
print(f"QA Model load time: {lap_time - start_time:.4f} seconds")
classifier_model, english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json = load_classifier_to_device()
end_time = time.time()
print(f"Classifier Model load time: {end_time - lap_time:.4f} seconds")

# blueprint
model_blueprint = Blueprint("Model Operations", __name__, url_prefix="/api/model", description="model operations")

@model_blueprint.route("/inference-device")
@model_blueprint.response(200)
def device():
    return jsonify({"device": str(torch_device())})

@model_blueprint.route("/test-classification", methods=["POST"])
@model_blueprint.arguments(ModelInputSchema)
@model_blueprint.response(200, ClassifierOutputSchema)
def classify(input):
    qs = input["question"]
    possibility_text, context_label = classifier_inference(qs, word_alias_dictionary, classifier_model, english_dictionary, english_list, train_categeory_data_json)
    return jsonify({"possibility": possibility_text, "category": context_label})

@model_blueprint.route("/test-qa", methods=["POST"])
@model_blueprint.arguments(ModelInputSchema)
@model_blueprint.response(200, ModelOutputSchema)
def qa(input):
    qs = input["question"]
    possibility_text, extracted_answer = hybrid_inference(qs, word_alias_dictionary, classifier_model, qa_model, qa_tokenizer, english_dictionary, english_list, train_categeory_data_json)
    return jsonify({"possibility": possibility_text, "answer": extracted_answer})