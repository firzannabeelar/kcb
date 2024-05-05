import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt


# preprocessing teks (tokenisasi, stopwords, dan stemming)

def preprocess_text(text):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()] 
    tokens = [stopword_remover.remove(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# sentiment analysis menggunakan Polarity Lexicon

def analyze_sentiment(text):
    positive_words = set(open("positive_words.txt", "r").read().splitlines())
    negative_words = set(open("negative_words.txt", "r").read().splitlines())
    tokens = word_tokenize(text)
    positive_count = sum(1 for word in tokens if word in positive_words)
    negative_count = sum(1 for word in tokens if word in negative_words)
    if positive_count > negative_count:
        return 'positif'
    elif positive_count < negative_count:
        return 'negatif'
    else:
        return 'netral'
    
#  topic modeling menggunakan Latent Dirichlet Allocation (LDA)

def perform_topic_modeling(data):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=None)
    tf = vectorizer.fit_transform(data)
    lda_model = LatentDirichletAllocation(n_components=68, learning_method='online', random_state=42, n_jobs=-1)
    lda_model.fit(tf)
    
    features = vectorizer.get_feature_names_out()
    
    topics = []
    
    for idx, topic in enumerate(lda_model.components_):
        top_features_indices = topic.argsort()[:-10 - 1:-1]
        top_features = [features[i] for i in top_features_indices]
        
        topics.append(top_features)
            
    return topics

def plot_topic_modeling(topics, sentiment):
    plt.figure(figsize=(10, 6))
    plt.barh(list(topics.keys()), list(topics.values()), color='skyblue')
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Kata')
    plt.title(f'Diagram Topic Modeling {sentiment.capitalize()}')
    plt.show()

# Membaca data ulasan dari file CSV

with open('google.csv', 'r', encoding='utf-8') as file:
    data = file.readlines()

# Membuat DataFrame dari data
    
df = pd.DataFrame(data, columns=['Ulasan'])

# Preprocessing teks

df['Preprocessed_Ulasan'] = df['Ulasan'].apply(preprocess_text)

# Analisis sentimen

df['Sentimen'] = df['Preprocessed_Ulasan'].apply(analyze_sentiment)
print("\nHasil Analisis Sentimen:\n", df[['Ulasan', 'Sentimen']])

# Topic modeling

topics = perform_topic_modeling(df['Preprocessed_Ulasan'])

print("\nTopik dari Ulasan UPN 'Veteran' Jawa Timur:")
for idx, features in enumerate(topics):
    print(features)


# Kesimpulan
    
positif_count = df['Sentimen'].value_counts().get('positif', 0)
negatif_count = df['Sentimen'].value_counts().get('negatif', 0)
netral_count = df['Sentimen'].value_counts().get('netral', 0)

print("\nKesimpulan:")
print("Jumlah ulasan positif:", positif_count)
print("Jumlah ulasan negatif:", negatif_count)
print("Jumlah ulasan netral:", netral_count)

if positif_count > negatif_count:
    print("Ulasan dalam dataset menunjukkan mayoritas sentimen POSITIF.")
elif positif_count < negatif_count:
    print("Ulasan dalam dataset menunjukkan mayoritas sentimen NEGATIF.")
else:
    print("Ulasan dalam dataset menunjukkan mayoritas sentimen NETRAL.")


# grafik topic ulasan
    
for sentiment in ['positif', 'negatif', 'netral']:
    sentiment_df = df[df['Sentimen'] == sentiment]
    preprocessed_data = sentiment_df['Preprocessed_Ulasan'].tolist()
    topics = perform_topic_modeling(preprocessed_data)
    
    # Flatten topics
    flattened_topics = {}
    for topic in topics:
        for word in topic:
            flattened_topics[word] = flattened_topics.get(word, 0) + 1
    
    # Sort flattened_topics by value (number of occurrences) in descending order
    sorted_topics = dict(sorted(flattened_topics.items(), key=lambda item: item[1], reverse=False))
    
    plot_topic_modeling(sorted_topics, sentiment)