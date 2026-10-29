TP : TF-IDF et similarité cosinus
Objectif
Ce projet permet de retrouver le document le plus proche d'une requête en utilisant :

TF-IDF
La similarité cosinus
Python
Scikit-learn
Requête principale
La requête recherchée est :

train billet réservation

Tableau des documents
Document	Contenu
D1	La réservation du billet de train se fait sur notre application mobile.
D2	Le billet de train peut être annulé depuis votre espace de réservation.
D3	Un abonnement de bus permet de voyager chaque jour dans la ville.
D4	Le service propose une réservation de chambre dans un hôtel proche de la gare.
D5	Pour réserver un billet de train, consultez les horaires puis confirmez votre réservation.
Tableau des résultats
Classement	Document	Similarité cosinus
1	D1	0.614659
2	D2	0.614659
3	D5	0.482595
4	D4	0.114591
5	D3	0.000000
Interprétation
Les documents D1 et D2 sont les plus proches de la requête.

Ils contiennent les trois mots importants :

train
billet
réservation
Le document D3 obtient un score de 0, car il ne contient aucun mot important de la requête.

Exécution du programme
Installer la bibliothèque :

pip install scikit-learn