# Skating Aerial Alignment

Projet Python pour explorer la phase aérienne d'un saut en patinage
artistique en fonction de l'alignement entre le moment cinétique et l'axe
longitudinal du corps.

Le projet fournit :

- un modèle multicorps `biorbd` simplifié du patineur ;
- une simulation de vol sans gravité pour la dynamique rotationnelle ;
- une interface graphique interactive ;
- une interface en ligne de commande pour lancer des campagnes de simulation ;
- une sauvegarde structurée des résultats pour créer un vrai laboratoire.

## Pour les débutants absolus

Si vous n'avez encore rien installé, le plus simple est d'utiliser
**Miniconda**.

### 1. Installer Miniconda

Source officielle :

- [Guide Miniconda officiel](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Guide macOS terminal officiel](https://www.anaconda.com/docs/getting-started/miniconda/install/mac-cli-install)
- [Page Miniconda officielle](https://www.anaconda.com/docs/getting-started/miniconda/main)

Si vous êtes sur un Mac Apple Silicon (`M1`, `M2`, `M3`, etc.), ouvrez
`Terminal` et lancez :

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh
```

Pendant l'installation :

1. appuyez sur `Entrée` pour faire défiler la licence ;
2. tapez `yes` pour accepter ;
3. acceptez le dossier par défaut sauf si vous savez exactement pourquoi le changer ;
4. répondez `yes` quand l'installateur propose d'initialiser `conda`.

Ensuite, **fermez complètement le Terminal** puis rouvrez-le.

Vérifiez que Miniconda fonctionne :

```bash
conda --version
```

Vous devriez voir quelque chose comme `conda 25.x.x`.

### 2. Créer un environnement Python pour ce projet

Dans le Terminal :

```bash
conda create -n skating-jump python=3.11 -y
conda activate skating-jump
```

Quand l'environnement est actif, vous voyez normalement `(skating-jump)` au
début de la ligne de commande.

### 3. Installer le projet

Placez-vous dans le dossier du projet :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
```

Puis installez les dépendances Python du projet :

```bash
python -m pip install -e .[test]
```

### 4. Installer `biorbd`

Le projet a besoin de `biorbd`. Comme son installation dépend souvent de
l'environnement scientifique déjà utilisé, gardez votre méthode habituelle si
vous en avez une.

Si `biorbd` n'est pas encore installé dans votre environnement, il faut
l'ajouter avant de lancer le projet.

### 5. Vérifier que tout marche

```bash
python -m pytest -q
```

Si les tests passent, l'environnement est prêt.

## Démarrage rapide

### Lancer la GUI

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python skating_aerial_alignment_gui.py
```

Alternative :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python -m skating_aerial_alignment gui
```

### Lancer une simulation unique

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python -m skating_aerial_alignment simulate \
  --output-dir artifacts/lab/run_demo \
  --sigma-rps 0.0 0.2 3.0 \
  --vertical-velocity 2.6 \
  --backward-velocity 1.5 \
  --inward-tilt-deg -8 \
  --print-summary
```

### Lancer un batch complet

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python -m skating_aerial_alignment batch \
  --config artifacts/lab/scenarios.json \
  --output-dir artifacts/lab/batch_001
```

### Comparer les résultats d'un batch

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python -m skating_aerial_alignment compare \
  --batch-dir artifacts/lab/batch_001 \
  --metric twist_turns
```

### Exporter les figures d'un run

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src python -m skating_aerial_alignment export-plots \
  --run-dir artifacts/lab/batch_001/run_001_baseline_backspin
```

## Ce que la CLI sauvegarde

Pour un run unique :

- `parameters.json`
- `summary.json`
- `timeseries.npz`

Pour un batch :

- un dossier par scénario ;
- `batch_summary.json`
- `batch_summary.csv`
- `comparison_<metric>.json`
- `comparison_<metric>.csv`
- `comparison_<metric>.png`

## Scénarios déjà prêts

Le dossier [artifacts/lab](/Users/mickaelbegon/Documents/Skating_jmp/artifacts/lab)
contient déjà :

- [scenarios.json](/Users/mickaelbegon/Documents/Skating_jmp/artifacts/lab/scenarios.json)
  : un batch de démonstration ;
- [README.md](/Users/mickaelbegon/Documents/Skating_jmp/artifacts/lab/README.md)
  : les commandes prêtes à lancer.

## Ce qu'on peut déjà étudier

Le prototype permet déjà de :

- choisir les composantes du moment cinétique dans le repère global ;
- exprimer ces composantes en équivalent rotations/s ;
- choisir la vitesse verticale via un temps de vol cible ;
- choisir une vitesse arrière du centre de masse ;
- choisir l'inclinaison initiale en salto et vers l'intérieur ;
- simuler un vol jusqu'au retour au sol ;
- stabiliser le tronc avec un contrôleur PD ;
- optimiser l'inclinaison intérieure pour maximiser la vrille ;
- comparer plusieurs scénarios entre eux ;
- exporter les séries temporelles et les figures.

## Idées de laboratoire à ajouter ensuite

Voici les axes les plus intéressants pour faire grossir le laboratoire :

- bibliothèques de protocoles prêtes à l'emploi ;
- balayages automatiques de paramètres ;
- exports automatiques de figures et de rapports ;
- comparaisons multi-critères ;
- optimisation batch des gains PD ;
- optimisation batch de l'inclinaison intérieure ;
- tableaux récapitulatifs de campagnes ;
- archivage de toutes les campagnes dans une base légère ;
- génération automatique de vidéos ;
- analyse de sensibilité ;
- profils de patineurs différents ;
- version plus riche du modèle biomécanique.

## Structure utile du dépôt

- [src/skating_aerial_alignment/modeling/biomod.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/modeling/biomod.py)
  : construction du modèle `biorbd`
- [src/skating_aerial_alignment/simulation/flight.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/simulation/flight.py)
  : simulation du vol
- [src/skating_aerial_alignment/cli.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/cli.py)
  : CLI du laboratoire
- [src/skating_aerial_alignment/visualization/app.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/visualization/app.py)
  : interface graphique
- [tests](/Users/mickaelbegon/Documents/Skating_jmp/tests)
  : tests unitaires
