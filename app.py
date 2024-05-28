from flask import Flask, render_template, request
app = Flask(__name__)
import os
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nrclex import NRCLex

packages = [
    'punkt',
    'averaged_perceptron_tagger',
    'vader_lexicon',
    'stopwords'
]

for package in packages:
    nltk.download(package, download_dir='/app/nltk_data')
        
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return f'File successfully uploaded and processed.'
    else:
        return 'Invalid file type. Only CSV files are allowed.'

def get_most_recent_file(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
        most_recent_file = max(files, key=os.path.getctime)
        return most_recent_file

def clean_uploads_folder(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    for f in files:
        os.remove(f)

def analyze_sentiment_from_recent_file(directory):
    csv_file = get_most_recent_file(UPLOAD_FOLDER)
    data = pd.read_csv(csv_file)
    argument = 'content'
    data = data.dropna(subset=[argument])
    sia = SentimentIntensityAnalyzer()
    plt.rc('font', size=6)
    def analyze_sentiment(description):
        scores = sia.polarity_scores(description)
        return scores

    data['Sentiment'] = data[argument].apply(analyze_sentiment)
    data['Compound'] = data['Sentiment'].apply(lambda d: d['compound'])
    data['Sentiment Category'] = pd.cut(data['Compound'], bins=[-1, -0.75, -0.2, 0.2, 0.75, 1], labels=['Vrlo negativan', 'Negativan', 'Neutralan', 'Pozitivan', 'Vrlo pozitivan'])

    sentiment_counts = data['Sentiment Category'].value_counts()

    colors = {
        'Vrlo negativan': 'maroon',
        'Negativan': 'red',
        'Neutralan': 'beige',
        'Pozitivan': 'lime',
        'Vrlo pozitivan': 'green'
    }
    fig, axes = plt.subplots(3, 2, figsize=(15, 8))
    for category, color in colors.items():
        subset = data[data['Sentiment Category'] == category]
        subset['Compound'].plot(kind='hist', bins=20, color=color, alpha=0.5, label=category, ax=axes[0, 0])

    axes[0, 0].set_title('Distribucija sentimenta')
    axes[0, 0].set_xlabel('Ocjena')
    axes[0, 0].set_ylabel('Količina')
    axes[0, 0].legend()

    axes[0, 1].pie(sentiment_counts, labels=sentiment_counts.index, colors=[colors[label] for label in sentiment_counts.index], autopct='%1.1f%%', startangle=140)
    axes[0, 1].set_title('Distribucija sentimenta (tortni graf)')

    def sentcateval(mean):
        category = ''
        if mean >= -1 and mean <= - 0.75 or mean < -1:
            category = '(Vrlo negativan)'
        elif mean >= -0.749999999 and mean <= -0.2:
            category = '(Negativan)'
        elif mean >= -0.199999999 and mean <= 0.2:
            category = '(Neutralan)'
        elif mean >= 0.200000001 and mean <= 0.75:
            category = '(Pozitivan)'
        elif mean >= 0.7500000001 and mean <= 1 or mean > 1:
            category = '(Vrlo pozitivan)'
        return category

    text_content = (
        'Srednja ocjena sentimenta: ' + str(round(data['Compound'].mean(),3)) + ' ' + sentcateval(data['Compound'].mean()) + '\n' +
        '68% sentimenata su između ocjene ' + str(round(data['Compound'].mean() - data['Compound'].std(),3)) + ' ' + sentcateval(data['Compound'].mean() - data['Compound'].std()) + 
        ' i ocjene ' + str(round(data['Compound'].mean() + data['Compound'].std(),3)) + ' ' + sentcateval(data['Compound'].mean() + data['Compound'].std()) + '\n' +
        'Najviši sentiment: ' + str(data['Compound'].max()) + '\n' +
        'Najniži sentiment: ' + str(data['Compound'].min())
    )

    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, text_content, ha='center', va='center', fontsize=12, wrap=True)

    data[argument] = data[argument].fillna('').astype(str)
    all_descriptions = ' '.join(data[argument].tolist())
    tokens = word_tokenize(all_descriptions)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tags = nltk.pos_tag(tokens)
    nouns = [word for word, pos in tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    freq_dist = FreqDist(nouns)
    most_common_nouns = freq_dist.most_common(10)

    words, counts = zip(*most_common_nouns)

    axes[1, 1].bar(words, counts, color='blue')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_ylabel('Učestalost')
    axes[1, 1].set_title('Najčešće riječi')
    axes[1, 1].tick_params(axis='x', rotation=45)

    text_object = NRCLex(all_descriptions)
    emotion_scores = text_object.raw_emotion_scores

    total_emotions = sum(emotion_scores.values())
    emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_scores.items()}

    custom_labels = {
        'fear': 'Strah',
        'anger': 'Frustracija',
        'anticipation': 'Uzbuđenost',
        'trust': 'Povjerenje',
        'surprise': 'Iznenađenje',
        'positive': 'Pozitiva',
        'negative': 'Negativa',
        'sadness': 'Tuga',
        'disgust': 'Gađenje',
        'joy': 'Sreća'
    }

    custom_emotion_percentages = {custom_labels[emotion]: value for emotion, value in emotion_percentages.items()}

    emotion_labels = list(custom_emotion_percentages.keys())
    emotion_values = list(custom_emotion_percentages.values())

    axes[2, 0].bar(emotion_labels, emotion_values, color=['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'grey'])
    axes[2, 0].set_xlabel('Emocije')
    axes[2, 0].set_ylabel('Postotak')
    axes[2, 0].set_title('Distribucija emocija')

    axes[2, 1].pie(emotion_values, labels=emotion_labels, autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'grey'])
    axes[2, 1].set_title('Distribucija emocija (Tortni graf)')

    plt.tight_layout()
    plt.show()

    clean_uploads_folder(UPLOAD_FOLDER)

@app.route('/analiza_sentimenta', methods=['POST'])
def run_script():
    # Run your script here
    analyze_sentiment_from_recent_file(UPLOAD_FOLDER)
    
    # Redirect back to home or display the result
    return render_template('index.html')  # or any other template you want to render


if __name__ == '__main__':
    app.run()
