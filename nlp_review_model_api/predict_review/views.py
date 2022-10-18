from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

# Create your views here.

from os.path import dirname, join, realpath
import joblib

from text_cleaning import text_cleaning

with open(
    join(dirname(realpath(__file__)),
         "..\models\sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


def index(request):

    template = loader.get_template('index.html')
    review = request.GET.get("review", "")

    if not review:
        return HttpResponse(template.render())

    # clean the review
    cleaned_review = text_cleaning(review)

    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))

    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    # show results
    result = {"prediction": sentiments[output],
              "probability": output_probability}

    return HttpResponse(template.render({"result": result}))
