
<!-- .slide: data-background="media/AI_Bootcamp.png" -->


# 公開！微軟的 RAG 新招式

Global AI Bootcamp 2025

2025/03/15 陳奎銘 Ben 


---




<!-- .slide: data-background-iframe="media/Ben.html" -->


---



<!-- .slide: data-auto-animate -->

# Outline
- RAG
- Azure AI Search
- GraphRAG
- RAG 一條龍


---

<!-- .slide: data-auto-animate -->
# RAG
RAG 的科普時間
- <font color='#646464'>Azure AI Search</font>
- <font color='#646464'>GraphRAG</font>
- <font color='#646464'>RAG 一條龍</font>



----

<font color='#FF2D2D'>Retrival</font>-<font color='#66B3FF'>Augmented</font> <font color='#79FF79'>Generation</font>
## 透過<font color='#FF2D2D'>檢索</font><font color='#66B3FF'>加強</font><font color='#79FF79'>生成</font>答案的能力

#### ⮕ 利用向量搜尋 <!-- .element: class="fragment" data-fragment-index="2" -->

----

#### `Cosine Similarity`

$$
 \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

![](media/cos_similarity.png)

----

### `Retrival-Augmented Generation`

- 結合『檢索』和『生成』，從外部知識中獲取答案
- 首先，將文件轉換成向量，存入向量資料庫
- 透過 cosine similarity 找到與問題最相近的片段
- 檢索 ⮕ 生成 ⮕ 回應



---

<!-- .slide: data-auto-animate -->
- <font color='#646464'>RAG</font>
# Azure AI `Search`
    - Semantic Rerank
    - Query Rewriting
    - Structure-Aware Chunking
    - Vector Compression
- <font color='#646464'>GraphRAG</font>
- <font color='#646464'>RAG 一條龍</font>




----

<!-- .slide: data-background="media/RAG_pipeline.png" -->


----

<!-- .slide: data-background="media/query_engine.png" -->


----


## `Semantic Ranker`

- Multi-lingual, deep learning models adapted from Microsoft Bing
- 3 steps to semantic ranking:
    - Collect and summarize inputs
    - Score results using the semantic ranker
    - Output rescored results, captions, and answers


----

### Collect & Summarize
- Select: top 50 results 
- List: "title","keyword", and "content" fields
- Trim: Up to 2048 tokens, where a token is approximately 10 characters.

----

### Score

- 4 ⮕ Perfectly answers the question 
- 3 ⮕ relevant but lacks details                          
- 2 ⮕ somewhat relevant; partially or some aspects        
- 1 ⮕ answers a small part of it                          
- 0 ⮕ irrelevant                                          


----



<!-- .slide: data-background="media/reranker_output.png" -->



----


<!-- .slide: data-auto-animate -->
## `Query Rerighting`

Query
```json [2|17-18]
{
  "search": "加班的規則",
  "count": true,
  "vectorQueries": [
    {
      "kind": "text",
      "text": "加班的規則",
      "fields": "contentVector",
      "queryRewrites": "generative"
    }
  ],
  "queryType": "semantic",
  "semanticConfiguration": "default",
  "captions": "extractive",
  "answers": "extractive|count-3",
  "queryLanguage": "en-us",
  "queryRewrites": "generative|count-5",
  "debug": "queryRewrites"
}
```


----

<!-- .slide: data-auto-animate -->

## Query Rerighting

Response

```json
"queryRewrites": {
      "text": {
        "inputQuery": "加班的規則",
        "rewrites": [
          "rules for overtime work",
          "regulations on overtime work",
          "regulations for overtime work",
          "rules for working overtime",
          "work overtime regulations"
        ]
      }
}
```

----

## `Structure Aware Chunking`
### <font color='orange'>`Markdown`</font>

----

<!-- .slide: data-background="media/structure_aware_chunking.png" -->


----

## `Vector Compression`

----

<!-- .slide: data-background="media/vector_compression.png" -->


----

<!-- .slide: data-background="media/MRL.png" -->



----

<!-- .slide: data-background="media/Scale_quality.png" -->



---


<!-- .slide: data-auto-animate -->
- <font color='#646464'>RAG</font>
- <font color='#646464'>Azure AI Search</font>
# `GraphRAG`


- <font color='#646464'>RAG 一條龍</font>



----

## RAG：讓 LLM 看書找答案

----

## `GraphRAG`
## 讓 LLM 產出資優生筆記

----

## `Knowledge` <font color='#FF2D2D'>`Graph`</font> + <font color='#66B3FF'>RAG</font>

- 文件 ⮕ 關係圖
    - 人、事、物 ⮕ Node (Entity)
    - 關係 ⮕ Edge (Relationship)
    - 小圈圈 ⮕ 很多 Node 群聚 (Community) 

----

<!-- .slide: data-background="media/graph_1.png" -->

----

<!-- .slide: data-background="media/graph_2.png" -->


----


<!-- .slide: data-background="media/community.png" -->

----


<!-- .slide: data-background="#999999" -->
GraphRAG Workflow From Microsoft
![](media/grapghRAG.svg)

----

## GraphRAG Indexing

![](media/index.png)


----

### Nodes

![](media/create_final_nodes.png)

----


### Relationships

![](media/create_final_relationships.png)


----


### Communities

![](media/create_final_community_reports.png)


----

## `Query`
- Global Search
- Local Search
- Drift Search


----

<!-- .slide: data-background-iframe="media/global.html" -->



----








<!-- .slide: data-background="media/global_search.png" -->

----

![](media/global_report.png)


----

- The trade war initiated by Trump has had significant repercussions for Asian economies, particularly those reliant on exports to the U.S. Countries like Taiwan and South Korea are navigating the challenges posed by increased tariffs and trade restrictions, which threaten their economic stability and growth prospects. This situation highlights the interconnectedness of global trade and the impact of U.S. policies on regional economies. [Data: Relationships (43, 2)]


----

- 川普發起的貿易戰對亞洲經濟產生了顯著影響，特別是那些依賴對美國出口的國家。例如，台灣和韓國正努力應對加徵關稅和貿易限制帶來的挑戰，這些挑戰威脅到了它們的經濟穩定與增長前景。此情況突顯了全球貿易的相互依存性以及美國政策對區域經濟的影響。 [Data：Relationships (43, 2)]

----



![](media/relationship.png)


----


<!-- .slide: data-background-iframe="media/local.html" -->


----


<!-- .slide: data-background="media/local_search.png" -->


----

### `Drift Search`

1. Global Search
2. Local Search
3. Output
    - 按相關性排序
    - 先給出主要結論，再列出支撐結論的各個要點


---

<!-- .slide: data-auto-animate -->

- <font color='#646464'>RAG</font>
- <font color='#646464'>Azure AI Search</font>
- <font color='#646464'>GraphRAG</font>
# RAG 一條龍
    - Youtube 爬蟲
    - Embedding Vector
    - Chatbot

----

## 實務上的 RAG 需求

![](media/rag_path.png)




----

## `Youtube`爬蟲


----

### Structured Outputs

```python
from pydantic import BaseModel

class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: str
    preparation: str

class Step(BaseModel):
    description: str

class Recipe(BaseModel):
    title: str
    ingredients: list[Ingredient]
    steps: Step
```


----

### 擷取 youtube 字幕

```python [6|32-36|92-105|111-126]
from urllib.parse import parse_qs
from urllib.parse import urlparse
from openai import OpenAI
from yt_dlp import YoutubeDL
import timeout_decorator
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import find_dotenv, load_dotenv


DEFAULT_LANGUAGES = ["zh-TW", "zh-Hant", "zh", "zh-Hans", "ja", "en", "ko"]


ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}


class YoutubeLoader:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or DEFAULT_LANGUAGES

    @timeout_decorator.timeout(20)
    def load(self, url: str) -> str:
 
        video_id = parse_video_id(url)

        # 擷取字幕
        transcript_pieces: list[dict[str, str | float]] = (
            YouTubeTranscriptApi().get_transcript(
                video_id, self.languages)
        )

        lines = []
        for transcript_piece in transcript_pieces:
            text = str(transcript_piece.get("text", "")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)


def parse_video_id(url: str) -> str:

    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise f"unsupported URL scheme: {parsed_url.scheme}"

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        raise f"unsupported URL netloc: {parsed_url.netloc}"

    path = parsed_url.path

    if path.endswith("/watch"):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if "v" in parsed_query:
            ids = parsed_query["v"]
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            raise f"no video found in URL: {url}"
    else:
        path = parsed_url.path.lstrip("/")
        video_id = path.split("/")[-1]

    if len(video_id) != 11:  # Video IDs are 11 characters long
        raise f"invalid video ID: {video_id}"

    return video_id


def get_youtube_videos(channel_url):

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    video_list = [
        {"title": entry["title"], "url": entry["url"]}
        for entry in info.get("entries", [])
    ]
    return video_list


# 定義食譜的內容
class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: str
    preparation: str

class Step(BaseModel):
    description: str

class Recipe(BaseModel):
    title: str
    ingredients: list[Ingredient]
    steps: Step

def main():

    load_dotenv(find_dotenv())

    client = OpenAI()
    yt_loader = YoutubeLoader()
    url = "https://youtu.be/smD_6Ranb4g"
    content = yt_loader.load(url)
    prompt = """
    從字幕中抽取食譜資訊，不要捏造任何資訊。
    抽取後請務必將所有內容翻譯成台灣繁體中文。"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"字幕：{content}"},
        ],
        response_format=Recipe,
    )
    markdown_recipe = completion.choices[0].message.content



if __name__ == "__main__":

    main()

```

----

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Recipe</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">title</span>=<span style="color: #008000; text-decoration-color: #008000">'和風鰤魚＋大根'</span>,
    <span style="color: #808000; text-decoration-color: #808000">ingredients</span>=<span style="font-weight: bold">[</span>
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'鰤魚粗'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">'清洗並去除腥味'</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'大根'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">'去皮並切成半月形'</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'長蔥'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">'切成白髮蔥與青蔥部分分開備用'</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'生薑'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">'清洗並切成薄片與細絲'</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'清酒'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'大量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">''</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'水'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">''</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'中皿糖'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">'主要用於去腥'</span><span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Ingredient</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'醬油'</span>, <span style="color: #808000; text-decoration-color: #808000">quantity</span>=<span style="color: #008000; text-decoration-color: #008000">'適量'</span>, <span style="color: #808000; text-decoration-color: #808000">unit</span>=<span style="color: #008000; text-decoration-color: #008000">''</span>, <span style="color: #808000; text-decoration-color: #808000">preparation</span>=<span style="color: #008000; text-decoration-color: #008000">''</span><span style="font-weight: bold">)</span>
    <span style="font-weight: bold">]</span>,
    <span style="color: #808000; text-decoration-color: #808000">steps</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Step</span><span style="font-weight: bold">(</span>
        <span style="color: #808000; text-decoration-color: #808000">description</span>=<span style="color: #008000; text-decoration-color: #008000">'1. 大根去皮並切成約2-3cm的半月切，備用。\n2. </span>
<span style="color: #008000; text-decoration-color: #008000">清洗鰤魚粗，去除表面血水，並以熱水燙過表面再立即用清水沖洗，去除腥味。\n3. 生薑切成兩種形式：薄片與細絲，備用。\n4.</span>
<span style="color: #008000; text-decoration-color: #008000">長蔥切成白髮蔥與青蔥部分分開備用，青蔥部分稍微壓碎以增香。\n5. 用寬型的鍋或平底鍋排放大根，加入清酒與水煮沸。\n6. </span>
<span style="color: #008000; text-decoration-color: #008000">將火調小，加入鰤魚粗，以及長蔥青蔥部分和生薑薄片同煮，以去腥。\n7. 加入中皿糖後繼續煮，去除浮沫。\n8. </span>
<span style="color: #008000; text-decoration-color: #008000">當煮沸後加入醬油，不蓋鍋蓋煮至水分減少，並不時將湯汁澆在食材上。\n9. 味道喜好程度煮至濃稠度適中，盛盤上桌。\n10. </span>
<span style="color: #008000; text-decoration-color: #008000">最後加入白髮蔥、生薑絲，喜好者可加入芥末增味，完成美味鰤大根。'</span>
    <span style="font-weight: bold">)</span>
<span style="font-weight: bold">)</span>
</pre>


----
## Embedding Vector

- Chunk
- Vector Database

----

### `Chunk` 

- 切分前先做成 <font color='orange'>`Markdown`</font> 格式
- 利用 <font color='orange'>`RecursiveCharacterTextSplitter`</font> 拆分
    - 依照 `separators` 拆分
        - `["#", "##", "###", "\n\n", "\n"]`
    - 再依照 `chunk_size` 拆分

----

### `Chunk` 範例

```python 
from langchain.docstore.document import Documente
from langchain.text_splitter import RecursiveCharacterTextSplitter

doc = Document(
    page_content=markdown_text, 
    metadata={"dataset": dataset, "file": markdown_file}
    )

markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\n\n", "\n", " "],
    chunk_size=1000,  # 可根據需求調整 chunk 大小
    chunk_overlap=100,  # 重疊區域，避免語境斷裂
)
documents = markdown_splitter.split_documents([doc])
```

----

### Vector Database
- Qdrant

----

<!-- .slide: data-background="media/qdrant.png" -->

----

### `Qdrant with Docker`

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```
http://localhost:6333/dashboard

----

<!-- .slide: data-background="media/qdrant_dashboard.png" -->

----

### `Embedding Sample Code`

```python [1-3|13-20 | 21-29]
doc = Document(
    page_content=markdown_text, 
    metadata={"dataset": dataset, "file": markdown_file}
)

markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\n\n", "\n", " "],
    chunk_size=1000,  # 可根據需求調整 chunk 大小
    chunk_overlap=100,  # 重疊區域，避免語境斷裂
)
documents = markdown_splitter.split_documents([doc])

from langchain_openai import AzureOpenAIEmbeddings
embedding_llm = AzureOpenAIEmbeddings(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_Embedding_DEPLOYMENT_NAME"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    )

from langchain_qdrant import QdrantVectorStore
qdrant = QdrantVectorStore.from_documents(
    documents,
    embedding=embedding_llm,
    url="http://localhost:6333",
    collection_name=collection,
)

```

----

<!-- .slide: data-background="media/qdrant_filter.png" -->


----

<!-- .slide: data-background="media/qdrant_filter.png" -->


```
metadata.dataset:"recipe"
```

----

### `Qdrant` 的注意事項

- 使用多個 collection 的代價就是系統資源的消耗
- 建議單一個 Collection 拆分給不同使用者使用
- 可以利用 Payload 切分資料



----

## `ChatBot`
- Streamlit
- Langchain + Qdrant

----

### 前端互動的頁面

### `streamlit`

----

### `streamlit sample code`

```python [1-3|5-9|11-18|20-39]
# set select box for dataset
dataset_name = st.sidebar.selectbox(
    "請選擇要查詢的資料集名稱", options=datasets)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = st.write_stream(
            get_response(
                user_query=user_query,
                chat_history=st.session_state.chat_history[-10:],
                collection_name="bootcamp",
                dataset_name=dataset_name,
            )
        )

    st.session_state.chat_history.append(
        AIMessage(content=response))


```

----

### 後端回答的機器人
### `Qdrant` + `Langchain`

----

### 後端回答的機器人

```python [4-17|18-26|27-33]
def get_response(
    user_query, chat_history, collection_name
):
    generator_llm = AzureChatOpenAI()
    system_prompt = (
        "你是一位專門根據文件回答問題的 AI 助手。"
        "如果你無法從文件得到答案，請說你不知道。"
        "請根據以下參考資料回答問題："
        "歷史紀錄：{chat_history}"
        "參考資料：{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(
        generator_llm, prompt)
    
    embedding_llm = AzureOpenAIEmbeddings()
    client = QdrantClient(url="http://localhost:6333")
    qdrant = QdrantVectorStore(
        client=client, 
        collection_name=collection_name, 
        embedding=embedding_llm
    )
    retriever = qdrant.as_retriever(
        search_kwargs={"k": 3})

    rag_chain = create_retrieval_chain(
        retriever, question_answer_chain)
    chain = rag_chain.pick("answer")
    return chain.stream({
        "input": user_query, 
        "chat_history": chat_history})


```



----

<!-- .slide: data-background="media/streamlit.png" -->


----


## Github Marketplace



----

### RAG for GitHub Models

<font size=3>from: https://techcommunity.microsoft.com/blog/azure-ai-services-blog/github-models-retrieval-augmented-generation-rag/4302518</font>

![](media/GH-Models-AZS-11-18-24.gif)




---

# 回家可以嘗試的項目
- 利用 Vector Search + Keyword Search + Query Rewriting 加強 RAG
- GraphRAG 可以提升 RAG 的全局觀
- 抓取 Youtube 字幕，快速了解影片內容
- Markdown 格式幾乎到處都用得到

----


# `Reference`

- <font size=6>Azure AI Search: https://pse.is/79b2bd</font>
- <font size=6>GrapahRAG: https://kuiming.github.io/graphrag-investing/output/</font>
- <font size=6>RAG 一條龍: https://kuiming.github.io/rag_path/output/</font>


----


## 投影片 + `Code`
![](media/QR_github.png)

----

## `Facebook`


<img src=media/QR_R_Ladies_Taipei.png width=50%></img><img src=media/QR_Ben_facebook.png width=50%></img>

----

<!-- .slide: data-background-iframe="https://www.accupass.com/go/legistaiwan_rladies" -->


----

<!-- .slide: data-background="media/legisTaiwan.png" -->



----

# Thank You