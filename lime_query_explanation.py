import fasttext
import re
import lime.lime_text
import numpy as np
from pathlib import Path


# Changes text into the same format as FastText classifier.
def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

# LIME needs to be able to mimic how the classifier splits
def tokenize_string(string):
    return string.split()


#-------------------------------------------------------#
#INPUTS THAT WILL NEED TO BE PUT IN. IT WORKS PEREFECTLY FINE WITH EXAMPLE DATA, BUT IF YOU WANT TO CUSTOMIZE THE RESULTS EITHER CHANGE THE MODEL OR REVIEW TEXT TO CHANGE RESULT.

nameofmodel = 'example'

review_to_test = "new owners have really given this place a facelift!  the veggie lavosh was recommended and did not disappoint!"


#-------------------------------------------------------#

# Load our trained FastText classifier model ()
classifier = fasttext.load_model(f'models/{nameofmodel}.bin')


# Create Lime explainer
explainer = lime.lime_text.LimeTextExplainer(

    split_expression=tokenize_string,

    # bow=False tells LIME to not assume that our classifier is based on single words only.
    bow=False,
    
    # To make the output pretty, tell LIME what to call each possible prediction from our model.
    class_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
)


def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []

    # This ensures we always get a probability score for every possible label in our model.
    labels, probabilities = classifier.predict(texts, 10)

    # returns predicitons sorted by most likely instead of in a fixed order.
    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs[order])

    return np.array(res)



# Pre-process the text of the review so it matches the training format
preprocessed_review = strip_formatting(review_to_test)

# Make a prediction and explain it!
exp = explainer.explain_instance(
    
    # The review to explain
    preprocessed_review,
    
    # The wrapper function that returns fasttext predictions in scikit-learn format
    classifier_fn=lambda x: fasttext_prediction_in_sklearn_format(classifier, x),
    
    # How many labels to explain
    top_labels=1,
    
    # How many words in our sentence to include in the explanation.
    num_features=20,

)

# Save the explanation to an HTML file so it's easy to view.
output_filename = Path(__file__).parent / "explanation.html"

#https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.explanation
# there are many different options: pyplot, list, map, etc...
exp.save_to_file(output_filename)