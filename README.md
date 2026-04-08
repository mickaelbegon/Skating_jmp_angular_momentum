# Skating Aerial Alignment

Projet Python pour explorer la phase aérienne d'un saut en patinage artistique
en fonction de l'alignement entre le moment cinétique et l'axe longitudinal du
corps.

L'application fournit :

- un modèle multicorps `biorbd` simplifié du patineur,
- une simulation de vol sans gravité pour la dynamique rotationnelle,
- une GUI `matplotlib` avec sliders, animation 3D et courbes temporelles,
- des tests unitaires et une CI GitHub Actions.

## Etat actuel du projet

Le prototype permet déjà de :

- choisir les composantes du moment cinétique dans le repère global ;
- exprimer les composantes en équivalent rotations/s dans la posture initiale ;
- choisir la vitesse verticale via un temps de vol cible ;
- choisir une vitesse arrière `V_arr` définie comme vitesse du centre de masse ;
- choisir l'inclinaison initiale en salto et vers l'intérieur ;
- simuler un vol jusqu'au retour au sol ;
- stabiliser le tronc avec un contrôleur PD et un réglage automatique ;
- optimiser l'inclinaison intérieure pour maximiser la vrille ;
- visualiser l'animation 3D, la trajectoire du CoM, l'angle entre `H` et l'axe
  longitudinal, la vrille, le salto, les 3 DoF du tronc et les efforts.

## Hypothèses principales

- Le bassin a 6 DoF flottants.
- Le tronc a 3 DoF de rotation par rapport au pelvis.
- Les autres segments sont figés dans une posture compacte de type backspin.
- La rotation est simulée sans gravité.
- La translation du centre de masse suit une trajectoire balistique imposée.
- La vitesse arrière agit sur le CoM et non sur le moment cinétique.

## Prérequis

- Python `>= 3.11`
- `numpy`, `scipy`, `matplotlib`
- `biorbd` pour la simulation et la GUI

Le projet déclare les dépendances Python standards dans
[pyproject.toml](/Users/mickaelbegon/Documents/Skating_jmp/pyproject.toml). `biorbd`
doit être disponible dans l'environnement utilisé pour lancer l'application.

L'interpréteur local utilisé pendant le développement est :

```bash
/Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python
```

## Installation

Installation editable :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
/Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m pip install -e .[test]
```

Si `biorbd` n'est pas déjà installé dans l'environnement, il faut l'ajouter à
part selon votre méthode habituelle.

## Lancement

Depuis la racine du projet :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python skating_aerial_alignment_gui.py
```

Alternative via le module :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m skating_aerial_alignment
```

## Utilisation rapide

Dans la GUI :

- `Hx`, `Hy`, `Hz` contrôlent le moment cinétique global sous forme
  d'équivalents rotations/s ;
- `Temps de vol (s)` pilote la vitesse verticale de décollage ;
- `V_arr (m/s)` pilote la vitesse arrière du centre de masse ;
- `Incl. salto (deg)` et `Incl. int. (deg)` règlent la posture initiale ;
- `Stabiliser le tronc` active la stabilisation PD avec tuning automatique ;
- `Avatar de face (vrille=0)` fige la coordonnée de vrille pour la vue 3D ;
- `Optimiser incl. interieure` cherche une inclinaison intérieure favorable à
  la vrille ;
- `Pause`, `Reset` et le slider temporel permettent de naviguer dans le vol ;
- `Vitesse 100% / 50% / 25%` ajuste la vitesse de lecture.

Les figures montrent notamment :

- l'angle entre le moment cinétique et l'axe longitudinal ;
- la vrille et le salto avec double axe vertical ;
- les 3 rotations du tronc ;
- les couples du tronc ;
- l'inertie apparente en vrille `||H|| / |omega_vrille|`.

## Tests et qualité

Suite complète :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m pytest -q
```

Vérifications de style :

```bash
cd /Users/mickaelbegon/Documents/Skating_jmp
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m black . --check
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m isort . --check-only --profile black
PYTHONPATH=src /Users/mickaelbegon/miniconda3/envs/vitpose-ekf/bin/python -m flake8 .
```

La CI est définie dans
[ci.yml](/Users/mickaelbegon/Documents/Skating_jmp/.github/workflows/ci.yml).

## Structure du dépôt

- [src/skating_aerial_alignment/modeling/biomod.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/modeling/biomod.py)
  : construction du modèle `biorbd`
- [src/skating_aerial_alignment/simulation/flight.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/simulation/flight.py)
  : simulation du vol, observables et optimisations simples
- [src/skating_aerial_alignment/visualization/app.py](/Users/mickaelbegon/Documents/Skating_jmp/src/skating_aerial_alignment/visualization/app.py)
  : interface graphique `matplotlib`
- [tests](/Users/mickaelbegon/Documents/Skating_jmp/tests)
  : tests unitaires et smoke tests GUI

## Limites actuelles

- Le modèle reste volontairement réduit.
- La stabilisation du tronc est basée sur un PD, pas sur une OCP complète.
- Les membres ne sont pas encore articulés indépendamment.
- L'interface est pensée d'abord pour l'exploration et le prototypage.
