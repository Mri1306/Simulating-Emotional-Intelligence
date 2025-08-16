
import json
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    classification_report,
    mean_squared_error
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from bert_score import score
import re
from tqdm import tqdm
import time
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TEST_DATA_PATH = r"E:\AI Chatbot LLM_Final\test_dataset.json"

EMOTION_CATEGORIES = [
    "happiness and joy",
    "fear & anxiety", 
    "love & affection",
    "emotional growth and self reflection",
    "neutral",
    "sadness and grief",
    "anger & frustration",
    "guilt and regret",
    "gratitude and contentment",
    "empathy and understanding others",
    "confidence and self belief",
    "decision-making"
]

EMOTION_MAP = {
    "joy": "happiness and joy",
    "excitement": "happiness and joy",
    "amusement": "happiness and joy",
    "pride": "confidence and self belief",
    "love": "love & affection",
    "admiration": "love & affection",
    "desire": "love & affection",
    "gratitude": "gratitude and contentment",
    "relief": "gratitude and contentment",
    "approval": "empathy and understanding others",
    "caring": "empathy and understanding others",
    "compassion": "empathy and understanding others",
    "sympathy": "empathy and understanding others",
    "optimism": "confidence and self belief",
    "hope": "confidence and self belief",
    "sadness": "sadness and grief",
    "grief": "sadness and grief",
    "disappointment": "sadness and grief",
    "embarrassment": "guilt and regret",
    "remorse": "guilt and regret",
    "guilt": "guilt and regret",
    "shame": "guilt and regret",
    "disgust": "guilt and regret",
    "anger": "anger & frustration",
    "annoyance": "anger & frustration",
    "disapproval": "anger & frustration",
    "frustration": "anger & frustration",
    "fear": "fear & anxiety",
    "nervousness": "fear & anxiety",
    "anxiety": "fear & anxiety",
    "worry": "fear & anxiety",
    "panic": "fear & anxiety",
    "confusion": "decision-making",
    "uncertainty": "decision-making",
    "realization": "emotional growth and self reflection",
    "insight": "emotional growth and self reflection",
    "surprise": "happiness and joy",
    "curiosity": "emotional growth and self reflection"
}

def map_emotion(emotion):
    """Map raw emotion to your defined emotion categories."""
    emotion = emotion.lower().strip()
    if emotion in EMOTION_CATEGORIES:
        return emotion
    elif emotion in EMOTION_MAP:
        return EMOTION_MAP[emotion]
    elif emotion == "neutral":
        return "neutral"
    else:
        print(f"Unknown emotion: {emotion}")
        return "other"

def load_test_data(file_path):
    """Load test data from JSON file and map emotions to your categories."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    target_emotions = []
    raw_emotions = []
    
    for item in data:
        match = re.search(r'Emotion: (.*?)\nA:', item['input'])
        if match:
            raw_emotion = match.group(1).strip()
            raw_emotions.append(raw_emotion)
            mapped_emotion = map_emotion(raw_emotion)
            target_emotions.append(mapped_emotion)
        else:
            raw_emotions.append("unknown")
            target_emotions.append("other")
    
    df = pd.DataFrame({
        'input': [item['input'] for item in data],
        'reference': [item['output'] for item in data],
        'raw_target_emotion': raw_emotions,
        'target_emotion': target_emotions
    })
    
    return df

def predict_emotions_zero_shot(texts, candidate_labels, batch_size=8):
    """Predict emotions using zero-shot classification pipeline."""
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size
    )
    
    hypothesis_template = "This text expresses {}"
    
    all_predictions = []
    raw_scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting emotions"):
        batch_texts = texts[i:i+batch_size]
        
        batch_results = classifier(
            batch_texts, 
            candidate_labels, 
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        
        for result in batch_results:
            predicted_emotion = result['labels'][0]
            all_predictions.append(predicted_emotion)
            scores_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
            raw_scores.append(scores_dict)
    
    return all_predictions, raw_scores

def apply_rule_based_corrections(df):
    """Apply rule-based corrections to emotion predictions."""
    corrected_emotions = df['predicted_emotion'].copy()
    
    for idx, row in df.iterrows():
        input_text = row['input'].lower()
        predicted = row['predicted_emotion']
        
        tone_match = re.search(r'Tone: (.*?)\n', input_text)
        body_match = re.search(r'Body Language: (.*?)\n', input_text)
        
        tone = tone_match.group(1).lower() if tone_match else ""
        body = body_match.group(1).lower() if body_match else ""
        
        if "smil" in body and predicted != "happiness and joy":
            if any(term in tone for term in ["positive", "cheerful", "happy"]):
                corrected_emotions.iloc[idx] = "happiness and joy"
        
        if any(term in body for term in ["cry", "tear", "sob"]) and predicted != "sadness and grief":
            corrected_emotions.iloc[idx] = "sadness and grief"
        
        if any(term in body for term in ["trembl", "shak", "nervous"]) and predicted != "fear & anxiety":
            corrected_emotions.iloc[idx] = "fear & anxiety"
        
        if "closed posture" in body and predicted not in ["fear & anxiety", "sadness and grief"]:
            corrected_emotions.iloc[idx] = "fear & anxiety"
        
        if "analytical" in tone and predicted not in ["neutral", "emotional growth and self reflection"]:
            corrected_emotions.iloc[idx] = "emotional growth and self reflection"
        
        if any(term in tone for term in ["aggress", "harsh", "direct"]) and any(term in body for term in ["frown", "clench", "tense"]):
            corrected_emotions.iloc[idx] = "anger & frustration"
        
        if any(term in tone for term in ["warm", "loving", "tender"]) and "open" in body:
            corrected_emotions.iloc[idx] = "love & affection"
        
        if any(term in tone for term in ["confident", "assertive"]) and any(term in body for term in ["straight", "upright", "eye contact"]):
            corrected_emotions.iloc[idx] = "confidence and self belief"
    
    return corrected_emotions

def get_emotion_embeddings(texts, sentence_model, batch_size=16):
    """Get embeddings for texts using sentence transformer."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = sentence_model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def plot_emotion_clusters(embeddings, emotions, predicted_emotions=None, method='pca', n_clusters=5):
    """Plot clustered emotion embeddings with centroids."""
    
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title_suffix = "PCA"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title_suffix = "t-SNE"
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    centroids_2d = kmeans.cluster_centers_
    
    
    unique_emotions = sorted(list(set(emotions)))
    emotion_colors = sns.color_palette("husl", len(unique_emotions))
    emotion_color_map = dict(zip(unique_emotions, emotion_colors))
    
    cluster_colors = sns.color_palette("Set1", n_clusters)
    
   
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        plt.scatter(reduced_embeddings[cluster_mask, 0], 
                   reduced_embeddings[cluster_mask, 1], 
                   c=[cluster_colors[i]], alpha=0.6, s=30, 
                   label=f'Cluster {i+1}')
    
    
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
               c='black', marker='x', s=200, linewidths=3, 
               label='Centroids')
    
    plt.title(f'K-Means Clustering ({title_suffix}) - {n_clusters} Clusters')
    plt.xlabel(f'{title_suffix} Component 1')
    plt.ylabel(f'{title_suffix} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    
    plt.subplot(2, 2, 2)
    for emotion in unique_emotions:
        emotion_mask = np.array(emotions) == emotion
        if np.any(emotion_mask):
            plt.scatter(reduced_embeddings[emotion_mask, 0], 
                       reduced_embeddings[emotion_mask, 1], 
                       c=[emotion_color_map[emotion]], alpha=0.7, s=30, 
                       label=emotion)
    
    plt.title(f'True Emotions ({title_suffix})')
    plt.xlabel(f'{title_suffix} Component 1')
    plt.ylabel(f'{title_suffix} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    
    if predicted_emotions is not None:
        plt.subplot(2, 2, 3)
        unique_pred_emotions = sorted(list(set(predicted_emotions)))
        pred_emotion_colors = sns.color_palette("husl", len(unique_pred_emotions))
        pred_emotion_color_map = dict(zip(unique_pred_emotions, pred_emotion_colors))
        
        for emotion in unique_pred_emotions:
            emotion_mask = np.array(predicted_emotions) == emotion
            if np.any(emotion_mask):
                plt.scatter(reduced_embeddings[emotion_mask, 0], 
                           reduced_embeddings[emotion_mask, 1], 
                           c=[pred_emotion_color_map[emotion]], alpha=0.7, s=30, 
                           label=emotion)
        
        plt.title(f'Predicted Emotions ({title_suffix})')
        plt.xlabel(f'{title_suffix} Component 1')
        plt.ylabel(f'{title_suffix} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    
    plt.subplot(2, 2, 4)
    
    
    cluster_emotion_counts = {}
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_emotions = np.array(emotions)[cluster_mask]
        emotion_counts = pd.Series(cluster_emotions).value_counts()
        cluster_emotion_counts[f'Cluster {i+1}'] = emotion_counts
    
    cluster_df = pd.DataFrame(cluster_emotion_counts).fillna(0)
    
    
    cluster_df.T.plot(kind='bar', stacked=True, ax=plt.gca(), 
                     color=[emotion_color_map.get(em, 'gray') for em in cluster_df.index])
    plt.title('Emotion Distribution by Cluster')
    plt.xlabel('Clusters')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'emotion_clusters_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, centroids_2d, reduced_embeddings

def calculate_cosine_similarity(generated_texts, reference_texts, sentence_model, batch_size=16):
    """Calculate cosine similarity between generated and reference texts."""
    similarities = []
    
    for i in tqdm(range(0, len(generated_texts), batch_size), desc="Calculating cosine similarity"):
        gen_batch = generated_texts[i:i+batch_size]
        ref_batch = reference_texts[i:i+batch_size]
        
        gen_embeddings = sentence_model.encode(gen_batch)
        ref_embeddings = sentence_model.encode(ref_batch)
        
        batch_similarities = []
        for gen_emb, ref_emb in zip(gen_embeddings, ref_embeddings):
            similarity = 1 - cosine(gen_emb, ref_emb)
            batch_similarities.append(similarity)
            
        similarities.extend(batch_similarities)
        
    return similarities

def calculate_bert_score(generated_texts, reference_texts, batch_size=8):
    """Calculate BERTScore between generated and reference texts."""
    all_P, all_R, all_F1 = [], [], []
    
    for i in tqdm(range(0, len(generated_texts), batch_size), desc="Calculating BERTScore"):
        gen_batch = generated_texts[i:i+batch_size]
        ref_batch = reference_texts[i:i+batch_size]
        
        P, R, F1 = score(gen_batch, ref_batch, lang="en", verbose=False)
        
        all_P.extend(P.tolist())
        all_R.extend(R.tolist())
        all_F1.extend(F1.tolist())
    
    return all_P, all_R, all_F1

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()
    
    return cm

def plot_metrics_by_emotion(metrics_df, title="Metrics by Emotion"):
    """Plot precision, recall, and F1 by emotion."""
    metrics = metrics_df[['precision', 'recall', 'f1-score']]
    metrics = metrics.reset_index().rename(columns={'index': 'emotion'})
    metrics = metrics.melt(id_vars=['emotion'], var_name='metric', value_name='value')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='emotion', y='value', hue='metric', data=metrics)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_emotion_distributions(df, title="Emotion Distributions"):
    """Plot target and predicted emotion distributions."""
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    target_counts = df['target_emotion'].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('Target Emotion Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    pred_counts = df['predicted_emotion'].value_counts()
    sns.barplot(x=pred_counts.index, y=pred_counts.values)
    plt.title('Predicted Emotion Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def calculate_perplexity(model, tokenizer, texts, batch_size=8):
    """Calculate perplexity for a list of texts."""
    perplexities = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
        batch_texts = texts[i:i+batch_size]
        batch_perplexities = []
        
        for text in batch_texts:
            encodings = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                batch_perplexities.append(perplexity)
        
        perplexities.extend(batch_perplexities)
    
    return perplexities

def evaluate_emotion_model(test_data_path, your_model_output_path=None, batch_size=8, use_rule_based=False, enable_clustering=True):
    """Evaluate emotion prediction using zero-shot classification with clustering visualization."""
    print("Loading test data...")
    df = load_test_data(test_data_path)
    
    if your_model_output_path:
        with open(your_model_output_path, 'r') as f:
            model_outputs = json.load(f)
        df['generated'] = model_outputs
    else:
        print("Using reference as generated for testing...")
        df['generated'] = df['reference']
    
    print("Predicting emotions with zero-shot classification...")
    df['predicted_emotion'], df['raw_scores'] = predict_emotions_zero_shot(
        df['generated'].tolist(), 
        EMOTION_CATEGORIES,
        batch_size=batch_size
    )
    
    if use_rule_based:
        print("Applying rule-based corrections...")
        df['predicted_emotion'] = apply_rule_based_corrections(df)
    
    print("Calculating emotion metrics...")
    
    all_emotions = sorted(list(set(df['target_emotion'].unique()) | set(df['predicted_emotion'].unique())))
    emotion_to_id = {e: i for i, e in enumerate(all_emotions)}
    
    y_true = [emotion_to_id[e] for e in df['target_emotion']]
    y_pred = [emotion_to_id.get(e, -1) for e in df['predicted_emotion']]
    
    accuracy = accuracy_score(y_true, y_pred)
    
    present_emotions = sorted(set(df['target_emotion']))
    present_emotion_ids = [emotion_to_id[e] for e in present_emotions]
    
    mse = mean_squared_error(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    
    report = classification_report(
        df['target_emotion'], 
        df['predicted_emotion'],
        labels=all_emotions,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    
    print("Calculating BERTScore...")
    P, R, F1 = calculate_bert_score(
        df['generated'].tolist(), 
        df['reference'].tolist(),
        batch_size=batch_size
    )
    df['bert_precision'] = P
    df['bert_recall'] = R
    df['bert_f1'] = F1
    
    print("Calculating cosine similarity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    df['cosine_similarity'] = calculate_cosine_similarity(
        df['generated'].tolist(), 
        df['reference'].tolist(), 
        sentence_model,
        batch_size=batch_size*2
    )
    
    
    if enable_clustering:
        print("Generating emotion embeddings for clustering...")
        embeddings = get_emotion_embeddings(df['generated'].tolist(), sentence_model, batch_size=batch_size*2)
        
        print("Creating clustered visualizations...")
        
        cluster_labels_pca, centroids_pca, reduced_pca = plot_emotion_clusters(
            embeddings, 
            df['target_emotion'].tolist(), 
            df['predicted_emotion'].tolist(), 
            method='pca', 
            n_clusters=len(EMOTION_CATEGORIES)
        )
        
        
        if len(df) <= 1000:  
            cluster_labels_tsne, centroids_tsne, reduced_tsne = plot_emotion_clusters(
                embeddings, 
                df['target_emotion'].tolist(), 
                df['predicted_emotion'].tolist(), 
                method='tsne', 
                n_clusters=len(EMOTION_CATEGORIES)
            )
        
        
        df['cluster_pca'] = cluster_labels_pca
        
      
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        silhouette_pca = silhouette_score(reduced_pca, cluster_labels_pca)
        
        # Compare clustering with true emotions
        emotion_labels = [emotion_to_id[e] for e in df['target_emotion']]
        ari_score = adjusted_rand_score(emotion_labels, cluster_labels_pca)
        
        print(f"Clustering Quality - Silhouette Score (PCA): {silhouette_pca:.4f}")
        print(f"Clustering vs True Emotions - Adjusted Rand Index: {ari_score:.4f}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading model for perplexity calculation...")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Calculating perplexity...")
        df['perplexity'] = calculate_perplexity(model, tokenizer, df['generated'].tolist(), batch_size=4)
        avg_perplexity = np.mean(df['perplexity'])
    except Exception as e:
        print(f"Warning: Perplexity calculation failed: {str(e)}")
        avg_perplexity = None
    
    print("Generating visualizations...")
    plot_emotion_distributions(df, "Emotion Distributions")
    
    try:
        cm = confusion_matrix(df['target_emotion'], df['predicted_emotion'], labels=present_emotions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=present_emotions, yticklabels=present_emotions)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Confusion matrix visualization failed: {str(e)}")
    
    plot_metrics_by_emotion(report_df.iloc[:-3], "Metrics by Emotion")
    
    results = {
        'emotional_accuracy': float(accuracy),
        'mean_squared_error': float(mse),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'average_bert_f1': float(np.mean(F1)),
        'average_cosine_similarity': float(np.mean(df['cosine_similarity']))
    }
    
    if enable_clustering:
        results['silhouette_score_pca'] = float(silhouette_pca)
        results['adjusted_rand_index'] = float(ari_score)
    
    if avg_perplexity is not None:
        results['average_perplexity'] = float(avg_perplexity)
    
    df.to_csv('detailed_results.csv', index=False)
    with open('summary_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('all_predictions.txt', 'w', encoding='utf-8') as f:
        f.write("All Predictions:\n==============================================================\n\n")
        for idx, row in df.iterrows():
            f.write(f"Sample #{idx+1}:\nInput: {row['input']}\nGenerated: {row['generated']}\n")
            f.write(f"Predicted Emotion: {row['predicted_emotion']}\nTrue Emotion: {row['target_emotion']}\n")
            if enable_clustering:
                f.write(f"PCA Cluster: {row['cluster_pca']}\n")
            f.write("--------------------------------------------------------------\n\n")
    
    print(f"\nEvaluation completed")
    print(f"Emotional Accuracy: {accuracy:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    if avg_perplexity is not None:
        print(f"Average Perplexity: {avg_perplexity:.3f}")
    print(f"Average BERTScore F1: {np.mean(F1):.4f}")
    print(f"Average Cosine Similarity: {np.mean(df['cosine_similarity']):.4f}")
    
    print("\nEmotion-wise Metrics:")
    for emotion in all_emotions:
        if emotion in report:
            metrics = report[emotion]
            print(f"{emotion}: F1={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    from collections import Counter
    print("\nPredicted Emotion Distribution:")
    print(Counter(df['predicted_emotion']))
    print("\nTrue Emotion Distribution:")
    print(Counter(df['target_emotion']))
    
    return results, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate emotion prediction model with clustering')
    parser.add_argument('--test_data', type=str, default=TEST_DATA_PATH, help='Path to test data JSON file')
    parser.add_argument('--model_outputs', type=str, default=None, help='Path to model outputs JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--use_rule_based', action='store_true', help='Apply rule-based corrections')
    parser.add_argument('--enable_clustering', action='store_true', default=True, help='Enable clustering analysis')
    
    args = parser.parse_args()
    args.test_data = os.path.normpath(args.test_data)
    if args.model_outputs:
        args.model_outputs = os.path.normpath(args.model_outputs)
    
    start_time = time.time()
    results, detailed_df = evaluate_emotion_model(
        args.test_data, 
        args.model_outputs, 
        batch_size=args.batch_size,
        use_rule_based=args.use_rule_based,
        enable_clustering=args.enable_clustering
    )
    end_time = time.time()
    
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
