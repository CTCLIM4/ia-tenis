# ia-tenis

Modelo de predicción de partidos de tenis (ATP/WTA) con Elo + features + calibración logística, más un analizador interactivo de value bets.

## Estado del proyecto

Personal research project, paper trading phase.

## Stack

- Python 3.14
- pandas, numpy, scikit-learn, scipy
- requests (descarga de datos ATP), openpyxl/xlrd (descarga de datos WTA)
- pytest

Dependencias completas en `requirements.txt`.

## Instalación

```bash
python -m venv tenis-env
./tenis-env/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source tenis-env/bin/activate && pip install -r requirements.txt  # Linux/Mac
```

## Uso

Descargar los datasets (no incluidos en el repo, ver más abajo):

```bash
./tenis-env/Scripts/python.exe scripts/download_data.py
```

Correr el pipeline completo (carga → features → backtest walk-forward) para un tour:

```bash
./tenis-env/Scripts/python.exe -m src.pipeline atp 1990 2026
./tenis-env/Scripts/python.exe -m src.pipeline wta 2007 2026
```

Analizador interactivo de value bets:

```bash
./tenis-env/Scripts/python.exe -m src.value_analysis          # ATP
./tenis-env/Scripts/python.exe -m src.value_analysis --wta    # WTA
```

Correr los tests:

```bash
./tenis-env/Scripts/python.exe -m pytest -v
```

## Fuentes de datos

- **ATP**: [stats.tennismylife.org](https://stats.tennismylife.org) (Tennismylife/TML-Database), licenciado bajo [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **WTA**: historial de [Tennis-Data](https://www.tennis-data.co.uk/data.php). Cuando el archivo de la temporada actual va atrasado, se complementa con partidos confirmados del cuadro principal de [Valuebetennis](https://www.valuebetennis.com/donnees.htm) (CC BY 4.0; atribución: *Valuebetennis, Résultats et cotes de tennis depuis 2021*). El complemento solo se usa si al menos el 90% de los partidos elegibles puede vincularse con jugadoras conocidas; los casos ambiguos se excluyen.
- **Cuotas** (opcional, para el analizador de value bets): [The Odds API](https://the-odds-api.com), vía la variable de entorno `ODDS_API_KEY`.

Ninguno de estos datasets se incluye en este repositorio — ver el aviso abajo.

## Aviso

Este repositorio **no incluye** datos descargados (`data/raw/`, `data/processed/`), modelos entrenados (`data/model_cache/`), ni logs de apuestas (`data/value_bets_log.csv`). Todo eso se genera localmente corriendo `scripts/download_data.py` y el pipeline — nunca se commitea (ver `.gitignore`).

## Licencia

Sin licencia — todos los derechos reservados. Repositorio privado, proyecto de investigación personal.
