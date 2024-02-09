from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class PredictParagraph(APIView):
    def post(self, request, *args, **kwargs):
        paragraphs = request.data.get('paragraphs', [])
        predicted_labels = self.predict_paragraphs(paragraphs)
        return Response({'predicted_labels': predicted_labels}, status=status.HTTP_200_OK)

    def predict_paragraphs(self, paragraphs):
    # Load pre-trained BERT model and tokenizer
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)

            # Define the paragraph types
            labels = ['legal', 'medical', 'educational','financial','educational','business','news','technical','creative writings','scientific research paper','government']

            # Tokenize and encode each paragraph separately
            tokenized_inputs = [tokenizer(p, padding=True, truncation=True, return_tensors="pt")['input_ids'] for p in paragraphs]

    # Perform inference for each tokenized input
            predicted_labels = []
            with torch.no_grad():
                for inputs in tokenized_inputs:
                    outputs = model(inputs)[0]
                    predicted_label_id = torch.argmax(outputs, dim=1).item()
                    predicted_label = labels[predicted_label_id]
                    predicted_labels.append(predicted_label)

            # Return predictions
            return predicted_labels
