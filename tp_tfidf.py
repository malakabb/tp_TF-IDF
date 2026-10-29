import json
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# ÉTAPE 1 : CHARGEMENT DU FICHIER JSON
# --------------------------------------------------

with open("dataset_tf_idf.json", "r", encoding="utf-8") as fichier:
    donnees = json.load(fichier)


# Récupération des documents
documents = donnees["documents"]

# Récupération de la requête principale
requete_principale = donnees["requete_principale"]

# Récupération des autres requêtes
autres_requetes = donnees["autres_requetes"]


# Création de deux listes :
# - une liste avec les identifiants D1, D2...
# - une liste avec les textes
identifiants = []
textes = []

for document in documents:
    identifiants.append(document["id"])
    textes.append(document["texte"])


# --------------------------------------------------
# ÉTAPE 2 : PRÉTRAITEMENT DES TEXTES
# --------------------------------------------------

# Liste simple de mots peu importants pour la recherche
mots_vides = [
    "la", "le", "les",
    "de", "du", "des",
    "un", "une",
    "dans", "sur",
    "notre", "votre",
    "se", "fait",
    "peut", "etre",
    "depuis", "chaque",
    "jour", "pour", "puis"
]


def nettoyer_texte(texte):
    """
    Cette fonction :
    1. met le texte en minuscules ;
    2. supprime les accents ;
    3. supprime la ponctuation.
    """

    # Passage en minuscules
    texte = texte.lower()

    # Suppression des accents
    texte = unicodedata.normalize("NFD", texte)

    texte = "".join(
        caractere
        for caractere in texte
        if unicodedata.category(caractere) != "Mn"
    )

    # Conservation des lettres, chiffres et espaces
    texte = re.sub(r"[^a-z0-9\s-]", " ", texte)

    return texte


# --------------------------------------------------
# ÉTAPE 3 : CONSTRUCTION DE L'ESPACE TF-IDF
# --------------------------------------------------

vectoriseur = TfidfVectorizer(
    preprocessor=nettoyer_texte,
    stop_words=mots_vides
)

# Apprentissage du vocabulaire sur les cinq documents
matrice_documents = vectoriseur.fit_transform(textes)

# Affichage du vocabulaire retenu
vocabulaire = vectoriseur.get_feature_names_out()

print("=" * 70)
print("VOCABULAIRE TF-IDF")
print("=" * 70)

print(vocabulaire)


# --------------------------------------------------
# ÉTAPE 4 : COMPARAISON D'UNE REQUÊTE
# --------------------------------------------------

def rechercher(requete):
    """
    Transforme une requête en vecteur TF-IDF,
    calcule la similarité cosinus et classe
    les documents du plus proche au moins proche.
    """

    print("\n" + "=" * 70)
    print("REQUÊTE :", requete)
    print("=" * 70)

    # Transformation de la requête en vecteur TF-IDF
    vecteur_requete = vectoriseur.transform([requete])

    # Cas où aucun mot de la requête n'existe dans le vocabulaire
    if vecteur_requete.nnz == 0:
        print("Aucune correspondance trouvée.")
        print("Les mots de la requête sont absents du vocabulaire.")
        return

    # Calcul de la similarité entre la requête et chaque document
    similarites = cosine_similarity(
        vecteur_requete,
        matrice_documents
    )[0]

    # Création de la liste des résultats
    resultats = []

    for index in range(len(documents)):
        resultat = {
            "id": identifiants[index],
            "texte": textes[index],
            "score": similarites[index]
        }

        resultats.append(resultat)

    # Classement par score décroissant
    resultats.sort(
        key=lambda resultat: resultat["score"],
        reverse=True
    )

    # Affichage des résultats
    print("\nCLASSEMENT DES DOCUMENTS\n")

    for position, resultat in enumerate(resultats, start=1):
        print(
            f"{position}. {resultat['id']} "
            f"- score : {resultat['score']:.6f}"
        )
        print(f"   {resultat['texte']}\n")

    # Meilleur document
    meilleur_document = resultats[0]

    if meilleur_document["score"] > 0:
        print("Meilleur document :", meilleur_document["id"])
        print(
            "Meilleur score :",
            f"{meilleur_document['score']:.6f}"
        )
    else:
        print("Aucun document ne correspond à la requête.")


# --------------------------------------------------
# ÉTAPE 5 : TEST DE LA REQUÊTE PRINCIPALE
# --------------------------------------------------

rechercher(requete_principale)


# --------------------------------------------------
# ÉTAPE 6 : TEST DES AUTRES REQUÊTES
# --------------------------------------------------

for requete in autres_requetes:
    rechercher(requete)