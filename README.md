# friendly-octo-pancake
Une app qui permet de différencier le vrai et le faux 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# 1. Simulation de données (Idéalement, chargez un fichier CSV)
data = {
    'text': [
        "Le soleil se lève à l'est.", 
        "Les chats peuvent voler dans l'espace.", 
        "La France est en Europe.", 
        "Manger des cailloux guérit le rhume."
    ],
    'label': ['REAL', 'FAKE', 'REAL', 'FAKE']
}
df = pd.DataFrame(data)

# 2. Séparation des données pour l'entraînement
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# 3. Transformation du texte en vecteurs numériques (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# 4. Initialisation et entraînement du classificateur
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 5. Test du modèle
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Précision du modèle : {round(score*100,2)}%')

# Fonction pour tester une nouvelle phrase
def verifier_info(phrase):
    vect = tfidf_vectorizer.transform([phrase])
    return pac.predict(vect)[0]

print(f"Résultat pour 'La lune est faite de fromage' : {verifier_info('La lune est faite de fromage')}")
