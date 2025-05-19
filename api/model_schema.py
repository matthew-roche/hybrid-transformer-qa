from marshmallow import Schema, fields, validate

class ModelInputSchema(Schema):
    question = fields.String(required=True, validate=validate.Length(min=5))

class ClassifierOutputSchema(Schema):
    possibility = fields.String()
    category = fields.String()

class ModelOutputSchema(Schema):
    possibility = fields.String()
    answer = fields.String()

