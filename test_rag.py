
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr
import torch
import json
import numpy as np
import random
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("Loading models...")


emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)


GO_TO_YOUR_15 = {
    "admiration": "confidence and self belief",
    "approval": "confidence and self belief",
    "gratitude": "gratitude and contentment",
    "pride": "confidence and self belief",
    "love": "love & affection",
    "joy": "happiness and joy",
    "amusement": "happiness and joy",
    "relief": "happiness and joy",
    "anger": "anger & frustration",
    "annoyance": "anger & frustration",
    "disapproval": "anger & frustration",
    "embarrassment": "guilt and regret",
    "nervousness": "fear & anxiety",
    "fear": "fear & anxiety",
    "sadness": "sadness and grief",
    "remorse": "guilt and regret",
    "confusion": "decision-making",
    "realization": "emotional growth and self reflection",
    "caring": "empathy and understanding others",
    "desire": "loneliness & isolation",
    "curiosity": "decision-making",
    "neutral": "neutral"
}


cosine_model = SentenceTransformer("all-MiniLM-L6-v2")


perplexity_model_name = "gpt2"
perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token


generation_model_name = "gpt2-large"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
generation_tokenizer.pad_token = generation_tokenizer.eos_token


emotion_labels = list(set(GO_TO_YOUR_15.values()))


true_emotions = []
pred_emotions = []
cosine_scores = []
bert_f1s = []
all_emotion_scores = []
all_true_emotion_one_hot = []
perplexity_scores = []

def calculate_perplexity(text):
    inputs = perplexity_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = perplexity_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
    return torch.exp(negative_log_likelihood).item()

def predict_emotion_roberta(text):
    result = emotion_classifier(text)[0][0]
    go_emotion = result['label'].lower()
    return GO_TO_YOUR_15.get(go_emotion, "other")


emotion_prompts = {
    "anger & frustration": "Answer the question with a tone of anger: {question} Respond in a way that expresses frustration or irritation.",
    "confidence and self belief": "Answer confidently, as if you strongly believe in yourself: {question}",
    "emotional growth and self reflection": "Reflect deeply and answer with self-awareness: {question}",
    "empathy and understanding others": "Answer with empathy and care for others' feelings: {question}",
    "fear & anxiety": "Answer with a tone of nervousness or concern: {question}",
    "gratitude and contentment": "Answer with gratitude and appreciation: {question}",
    "guilt and regret": "Respond with a tone of guilt or remorse: {question}",
    "happiness and joy": "Answer cheerfully and with happiness: {question}",
    "love & affection": "Respond warmly, showing love and care: {question}",
    "neutral": "Answer in a neutral and objective tone: {question}",
    "other": "Answer in a thoughtful and general tone: {question}",
    "sadness and grief": "Answer with sadness and emotional weight: {question}",
    "decision-making": "Answer in a deliberate, logical, and clear tone: {question}",
    "stress and pressure": "Respond as if under pressure or stress: {question}",
    "non-verbal communication": "Describe your thoughts while reflecting subtle emotional cues: {question}"
}


few_shot_examples = {
    "anger & frustration": [
        "I'm so tired of dealing with this incompetence! It's absolutely infuriating.",
        "This is ridiculous! I've been waiting for hours and nobody has helped me.",
        "Every time I try to make progress, someone throws another obstacle in my way. It's maddening!"
    ],
    "confidence and self belief": [
        "I know I can handle this challenge. I've prepared thoroughly and I'm ready.",
        "I've overcome similar situations before, and I'll succeed at this too.",
        "There's no doubt in my mind that I have the skills needed to excel here."
    ],
    "emotional growth and self reflection": [
        "Looking back, I can see how much I've changed and grown from that experience.",
        "I've learned to recognize my emotional patterns and work with them rather than against them.",
        "This situation has taught me a lot about myself and how I respond to challenges."
    ],
    "empathy and understanding others": [
        "I can see why you'd feel that way. It must be really difficult to go through that.",
        "Your perspective makes perfect sense given what you've experienced.",
        "I'm trying to put myself in your shoes and understand how this affects you."
    ],
    "fear & anxiety": [
        "I'm really worried about what might happen. What if everything goes wrong?",
        "My heart is racing just thinking about this situation. I'm not sure I can handle it.",
        "I keep imagining all the possible bad outcomes and it's making me so nervous."
    ],
    "gratitude and contentment": [
        "I'm so thankful for all the support I've received. It's made such a difference.",
        "I feel truly blessed to have these opportunities and experiences in my life.",
        "I'm content with where I am right now and appreciate all the good things I have."
    ],
    "guilt and regret": [
        "I wish I had handled that differently. I feel terrible about how it turned out.",
        "I can't stop thinking about my mistake and how it affected everyone.",
        "I should have known better and made a different choice. I regret what I did."
    ],
    "happiness and joy": [
        "I'm absolutely thrilled about this opportunity! It's exactly what I've been hoping for.",
        "This is wonderful news! I can't stop smiling thinking about it.",
        "I'm so happy right now, everything feels perfect and full of possibility!"
    ],
    "love & affection": [
        "I care about you deeply and want only the best for you.",
        "The connection we share means everything to me.",
        "My heart feels so full when I think about how much I care for you."
    ],
    "neutral": [
        "The data indicates several possible interpretations of the results.",
        "There are multiple factors to consider when evaluating this situation.",
        "From an objective standpoint, there are both advantages and disadvantages to this approach."
    ],
    "other": [
        "I have a variety of thoughts on this topic that don't fit neatly into one category.",
        "My perspective on this is complex and multifaceted.",
        "There are several dimensions to consider here beyond the obvious ones."
    ],
    "sadness and grief": [
        "I feel a deep sense of loss when I think about what happened.",
        "It's hard to imagine moving forward without them. The pain is overwhelming.",
        "I'm struggling to find joy in things I used to love. Everything feels empty now."
    ],
    "decision-making": [
        "After carefully weighing all options, I've determined this is the optimal course of action.",
        "The logical conclusion, based on the available evidence, points to this solution.",
        "I've analyzed the pros and cons thoroughly and reached a clear decision."
    ],
    "stress and pressure": [
        "I have so many deadlines and not enough time. I don't know how I'll get it all done.",
        "The expectations are overwhelming and I feel like I'm constantly racing against the clock.",
        "There's so much riding on this and I can feel the pressure mounting."
    ],
    "non-verbal communication": [
        "I notice my shoulders tensing as I consider this question. It's bringing up some discomfort.",
        "I find myself nodding as I think about this. It resonates with my experience.",
        "My expression probably shows my confusion right now. I'm trying to process this."
    ]
}


def create_example_database():
    """Create a database of example responses for each emotion"""
    database = []
    for emotion, examples in few_shot_examples.items():
        for example in examples:
            database.append({
                "text": example,
                "emotion": emotion,
                "embedding": cosine_model.encode(example)
            })
    return database


def retrieve_examples(query, target_emotion, database, k=3):
    """
    Retrieve examples that are semantically similar to the query and match the target emotion
    Using a combination strategy as described in search result [9]
    """
    query_embedding = cosine_model.encode(query)
    
    
    semantic_scores = []
    for item in database:
        semantic_score = cosine_similarity([query_embedding], [item["embedding"]])[0][0]
        semantic_scores.append(semantic_score)
    
    
    emotion_scores = [1.0 if item["emotion"] == target_emotion else 0.0 for item in database]
    
    
    combined_scores = [s * e for s, e in zip(semantic_scores, emotion_scores)]
    
    
    if sum(combined_scores) == 0:
        combined_scores = semantic_scores
    
    
    top_indices = np.argsort(combined_scores)[-k:][::-1]
    
    
    return [database[i]["text"] for i in top_indices]


def create_few_shot_prompt(question, target_emotion, retrieved_examples):
    """Create a few-shot prompt for emotion-conditioned generation"""
    
    prompt = emotion_prompts[target_emotion].format(question=question) + "\n\n"
    
    
    prompt += "Here are some examples of responses with a similar emotional tone:\n\n"
    for i, example in enumerate(retrieved_examples):
        prompt += f"Example {i+1}: {example}\n\n"
    
    
    prompt += f"Now, please answer the question '{question}' with a {target_emotion} tone:"
    
    return prompt


def generate_response(prompt, max_length=150):
    """Generate a response using the few-shot prompt"""
    inputs = generation_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=generation_tokenizer.eos_token_id
        )
    
   
    generated_text = generation_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def emotion_conditioned_rag(question, target_emotion, database):
    """
    Implement emotion-conditioned RAG:
    1. Retrieve examples based on semantic and emotional similarity
    2. Create a few-shot prompt with the retrieved examples
    3. Generate a response using the few-shot prompt
    """
    retrieved_examples = retrieve_examples(question, target_emotion, database)
    
    
    prompt = create_few_shot_prompt(question, target_emotion, retrieved_examples)
    
    
    response = generate_response(prompt)
    
    return response

print("Creating example database...")
example_database = create_example_database()

print("Loading test data...")
with open("test_dataset.json", "r") as f:
    test_data = json.load(f)

print(f"\nTesting on {len(test_data)} samples...\n")


detailed_sample_results = []

for i, sample in enumerate(test_data):
    try:
        question = sample["input"].split("Q:")[1].split("\n")[0].strip()
        emotion = sample["input"].split("Emotion:")[1].split("\n")[0].strip().lower()
        reference = sample["output"].strip()

        
        prediction = emotion_conditioned_rag(question, emotion, example_database)

        
        predicted_emotion = predict_emotion_roberta(prediction)
        pred_emotions.append(predicted_emotion)
        true_emotions.append(emotion)

        
        pred_vec = cosine_model.encode(prediction)
        ref_vec = cosine_model.encode(reference)
        cos = cosine_similarity([pred_vec], [ref_vec])[0][0]
        cosine_scores.append(cos)

        
        _, _, f1 = score([prediction], [reference], lang="en", verbose=False)
        bert_f1s.append(f1[0].item())

        
        perplexity_score = calculate_perplexity(prediction)
        perplexity_scores.append(perplexity_score)

        
        sample_result = {
            "sample_id": i + 1,
            "question": question,
            "target_emotion": emotion,
            "reference_response": reference,
            "generated_response": prediction,
            "predicted_emotion": predicted_emotion,
            "emotion_match": emotion == predicted_emotion,
            "metrics": {
                "bertscore_f1": f1[0].item(),
                "cosine_similarity": cos,
                "perplexity": perplexity_score
            }
        }
        detailed_sample_results.append(sample_result)

        print(f"Q{i+1}: {question}")
        print(f"Target: {emotion} | Predicted: {predicted_emotion}")
        print(f"Prediction: {prediction}")
        print(f"F1: {f1[0].item():.3f} | Cos: {cos:.3f} | PPL: {perplexity_score:.2f}")
        print("-" * 60)

    except Exception as e:
        print(f"Error in sample {i+1}: {e}")


emotion_mapping = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
true_vals = [emotion_mapping.get(e, -1) for e in true_emotions]
pred_vals = [emotion_mapping.get(e, -1) for e in pred_emotions]

pearson_corr, p_value = pearsonr(true_vals, pred_vals) if len(true_vals) > 1 else (0, 1)
mse = mean_squared_error(
    [[1.0 if emotion_mapping.get(e, -1) == i else 0 for i in range(len(emotion_labels))] for e in true_emotions],
    [[1.0 if emotion_mapping.get(e, -1) == i else 0 for i in range(len(emotion_labels))] for e in pred_emotions]
)
avg_perplexity = sum(perplexity_scores) / len(perplexity_scores) if perplexity_scores else 0


from sklearn.metrics import f1_score
emotion_f1_scores = {}
for emotion in emotion_labels:
    emotion_true = [1 if e == emotion else 0 for e in true_emotions]
    emotion_pred = [1 if e == emotion else 0 for e in pred_emotions]
    if sum(emotion_true) > 0:  
        emotion_f1_scores[emotion] = f1_score(emotion_true, emotion_pred, average='binary')

print("\n==== Final Evaluation ====")
print(f"Emotional Accuracy: {accuracy_score(true_emotions, pred_emotions) * 100:.2f}%")
print(f"Average BERTScore F1: {sum(bert_f1s)/len(bert_f1s):.3f}")
print(f"Average Cosine Similarity: {sum(cosine_scores)/len(cosine_scores):.3f}")
print(f"Pearson Correlation: {pearson_corr:.3f} (p-value: {p_value:.3f})")
print(f"Mean Squared Error: {mse:.3f}")
print(f"Average Perplexity: {avg_perplexity:.3f}")


evaluation_results = {
    "timestamp": datetime.now().isoformat(),
    "evaluation_metadata": {
        "total_samples": len(test_data),
        "successful_evaluations": len(detailed_sample_results),
        "emotion_labels": emotion_labels,
        "models": {
            "emotion_classifier": "j-hartmann/emotion-english-distilroberta-base",
            "cosine_similarity_model": "all-MiniLM-L6-v2",
            "perplexity_model": perplexity_model_name,
            "generation_model": generation_model_name
        }
    },
    "overall_metrics": {
        "emotional_accuracy_percent": accuracy_score(true_emotions, pred_emotions) * 100,
        "average_bertscore_f1": sum(bert_f1s)/len(bert_f1s) if bert_f1s else 0,
        "average_cosine_similarity": sum(cosine_scores)/len(cosine_scores) if cosine_scores else 0,
        "pearson_correlation": pearson_corr,
        "pearson_p_value": p_value,
        "mean_squared_error": mse,
        "average_perplexity": avg_perplexity
    },
    "emotion_specific_metrics": {
        "f1_scores_by_emotion": emotion_f1_scores
    },
    "detailed_sample_results": detailed_sample_results
}


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"rag_evaluation_results_{timestamp}.json"

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"\n Evaluation results saved to: {output_filename}")
print(f" Total samples evaluated: {len(detailed_sample_results)}")
print(f" Overall accuracy: {evaluation_results['overall_metrics']['emotional_accuracy_percent']:.2f}%")
