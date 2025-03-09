# Global AI Bootcamp 2025

## Slides
### Taichung
https://kuiming.github.io/rag_path/output/#/
### Taipei
https://kuiming.github.io/rag_path/output/microsoft_rag.html

## Code
### Please prepare the .env file

```bash
OPENAI_API_KEY="<your key>"
# Set your dataset names in Qdrant for RAG
QDRANT_DATASETS="datasetname1,datasetname2....."
```
### Packages

```bash
pip install requirements.txt
```

### Get recipe from youtube for RAG

- `youtube_recipe.py`: get recipe from youtube
- `rag.py`: tranform recipe into embedding vector
- `bot.py`: create chatbot with `streamlit`
```bash
mkdir recipe
python3 youtube_recipe.py
python3 rag.py
python3 -m streamlit run bot.py
```

### Crawler with structured output
- `espn_openai.py`: get commentaries from ESPN

### Get youtube summary
[youtube_abstracter.ipynb](https://github.com/KuiMing/rag_path/blob/main/youtube_abstracter.ipynb)

## Acknowledgements
This project was inspired by [iamlazy](https://github.com/narumiruna/iamlazy).
