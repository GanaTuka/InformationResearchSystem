import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import requests
from bs4 import BeautifulSoup
import time
import re
import os
from collections import defaultdict, Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

# DOCUMENT ЦУГЛУУЛАХ ХЭСЭГ 

class DocumentCollection:
    def __init__(self):
        self.documents = []
        self.inverted_index = defaultdict(list)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.scraping_log = []
        
    def scrape_website(self, url):
        """Scrape using requests - NO SELENIUM"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            self.scraping_log.append(f"📥 Trying: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            initial_count = len(self.documents)

            articles = soup.find_all('article')
            if articles:
                self.scraping_log.append(f"✓ Found {len(articles)} articles")
                for article in articles[:20]:
                    text = article.get_text(separator=' ', strip=True)
                    if len(text) > 100:
                        self.documents.append({
                            'id': len(self.documents),
                            'title': f"Document {len(self.documents)+1}",
                            'content': text,
                            'url': url,
                            'source': 'web_scrape'
                        })
            
            if len(self.documents) == initial_count:
                paragraphs = soup.find_all('p')
                self.scraping_log.append(f"✓ Олсон {len(paragraphs)} paragraphs")
                for p in paragraphs[:30]:
                    text = p.get_text(strip=True)
                    if len(text) > 100:
                        self.documents.append({
                            'id': len(self.documents),
                            'title': f"Document {len(self.documents)+1}",
                            'content': text,
                            'source': 'web_scrape'
                        })
            
            docs_added = len(self.documents) - initial_count
            if docs_added > 0:
                self.scraping_log.append(f"✅ Ачаалласан {docs_added} баримт мэдээлэл")
            else:
                self.scraping_log.append(f"❌ Контент олдсонгүй")
            return docs_added
        except Exception as e:
            self.scraping_log.append(f"❌ Алдаа: {str(e)}")
            return 0
    
    def add_sample_documents(self):
        samples = [
            "Information retrieval is the science of searching for information in documents and databases.",
            "Machine learning algorithms can improve search engine relevance and ranking.",
            "Natural language processing helps computers understand human language and text.",
            "TF-IDF measures term importance by frequency and inverse document frequency.",
            "Vector space models represent documents as vectors in high dimensional space.",
            "Boolean retrieval uses AND OR NOT operators for precise query matching.",
            "PageRank algorithm ranks web pages based on link structure and importance.",
            "Precision and recall are key metrics for evaluating information retrieval systems.",
            "Query expansion improves search results by adding related terms to queries.",
            "Inverted indexes enable fast full-text search across large document collections."
        ]
        for idx, text in enumerate(samples):
            self.documents.append({
                'id': len(self.documents),
                'title': f"Sample Document {idx+1}",
                'content': text,
                'source': 'sample'
            })
    
    def preprocess_text(self, text):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 
                    'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 
                    'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 
                    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        return [w for w in words if len(w) > 2 and w not in stopwords]
    
    def build_inverted_index(self):
        self.inverted_index = defaultdict(list)
        for doc in self.documents:
            tokens = self.preprocess_text(doc['content'])
            for position, token in enumerate(tokens):
                self.inverted_index[token].append((doc['id'], position))
        print(f"Built inverted index with {len(self.inverted_index)} unique terms")
    
    def build_tfidf_matrix(self):
        if not self.documents:
            return
        corpus = [doc['content'] for doc in self.documents]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200, stop_words='english', ngram_range=(1,2), min_df=1
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        print(f"Built TF-IDF matrix: {self.tfidf_matrix.shape}")


# SEARCH ENGINE ТОХИРГООНИЙ ЯДАРГААТАЙ ХЭСЭГ

class SearchEngine:
    def __init__(self, doc_collection):
        self.collection = doc_collection
        
    def boolean_search(self, query):
        terms = self.collection.preprocess_text(query)
        if not terms or terms[0] not in self.collection.inverted_index:
            return []
        result_docs = set(doc_id for doc_id, _ in self.collection.inverted_index[terms[0]])
        for term in terms[1:]:
            if term in self.collection.inverted_index:
                term_docs = set(doc_id for doc_id, _ in self.collection.inverted_index[term])
                result_docs = result_docs.intersection(term_docs)
            else:
                return []
        return list(result_docs)
    
    def vector_space_search(self, query, top_k=10):
        if self.collection.tfidf_matrix is None:
            return []
        query_vector = self.collection.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.collection.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'doc_id': idx,
                    'score': float(similarities[idx]),
                    'title': self.collection.documents[idx]['title'],
                    'content': self.collection.documents[idx]['content'][:200] + "..."
                })
        return results
    
    def query_expansion_rocchio(self, query, relevant_docs, alpha=1.0, beta=0.75):
        if not relevant_docs or self.collection.tfidf_matrix is None:
            return query
        query_vector = self.collection.tfidf_vectorizer.transform([query]).toarray()[0]
        relevant_vectors = self.collection.tfidf_matrix[relevant_docs].toarray()
        relevant_centroid = np.mean(relevant_vectors, axis=0)
        new_query_vector = alpha * query_vector + beta * relevant_centroid
        feature_names = self.collection.tfidf_vectorizer.get_feature_names_out()
        top_indices = new_query_vector.argsort()[-10:][::-1]
        expanded_terms = [feature_names[i] for i in top_indices if new_query_vector[i] > 0]
        return " ".join(expanded_terms[:5])


# EVALUATOR ХИЙХ

class Evaluator:
    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k).intersection(set(relevant)))
        return relevant_retrieved / k if k > 0 else 0
    
    @staticmethod
    def recall_at_k(retrieved, relevant, k):
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k).intersection(set(relevant)))
        return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0
    
    @staticmethod
    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(retrieved, relevant):
        if not relevant:
            return 0
        score, num_hits = 0.0, 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / len(relevant)
    
    @staticmethod
    def ndcg_at_k(retrieved, relevant, k):
        def dcg(scores, k):
            return sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))
        retrieved_at_k = retrieved[:k]
        relevance_scores = [1 if doc in relevant else 0 for doc in retrieved_at_k]
        ideal_scores = sorted(relevance_scores, reverse=True)
        dcg_value = dcg(relevance_scores, k)
        idcg_value = dcg(ideal_scores, k)
        return dcg_value / idcg_value if idcg_value > 0 else 0


# COMPONENT ХЭСЭГ 

doc_collection = DocumentCollection()
search_engine = None
evaluator = Evaluator()
initialization_done = False

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Мэдээллийн хайлтын систем By - Ganaa", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P("Индексжүүлэлт, Жагсаалт, Query expansion, and evaluation Үзүүлэлтүүдийг хамруулсан IR систем.",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    html.Div([
        html.H3("🔧 Алхам 1: Cистемийн ачааллах"),
        html.Button("Document-г ачааллах & Index үүсгэх", id='init-button', 
                   style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer'}),
        html.Div(id='init-status', style={'marginTop': '10px', 'color': '#27ae60'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("🔍 Алхам 2: Баримтийг хайх"),
        dcc.Input(id='query-input', type='text', placeholder='Enter your search query...',
                 style={'width': '60%', 'padding': '10px', 'fontSize': '14px'}),
        html.Button("Search", id='search-button', 
                   style={'marginLeft': '10px', 'padding': '10px 20px', 'cursor': 'pointer'}),
        
        html.Div([
            html.Label("Ашиглах аргууд:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='search-method',
                options=[
                    {'label': ' Boolean (AND)', 'value': 'boolean'},
                    {'label': ' Vector Space (TF-IDF)', 'value': 'vector'}
                ],
                value='vector',
                inline=True
            )
        ], style={'marginTop': '15px'}),
        
        html.Div(id='search-results', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("🎯 Алхам 3: Query Expansion (Rocchio)"),
        dcc.Input(id='relevant-docs', type='text', placeholder='Enter relevant doc IDs (comma-separated, e.g., 0,2,5)',
                 style={'width': '60%', 'padding': '10px'}),
        html.Button("Expand Query", id='expand-button', 
                   style={'marginLeft': '10px', 'padding': '10px 20px', 'cursor': 'pointer'}),
        html.Div(id='expanded-query', style={'marginTop': '10px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("📊 Алхам 4: Үнэлгээний үзүүлэлтүүд"),
        html.P("Одоогийн асуулгад үндэслэлтэй холбоотой баримт бичгийн дугаарыг оруулна уу:"),
        dcc.Input(id='ground-truth', type='text', placeholder='Relevant doc IDs (e.g., 0,1,3)',
                 style={'width': '60%', 'padding': '10px'}),
        html.Button("Calculate Metrics", id='eval-button', 
                   style={'marginLeft': '10px', 'padding': '10px 20px', 'cursor': 'pointer'}),
        html.Div(id='metrics-display', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("🧪 Алхам 5: Туршилтыг эхлүүлэх"),
        html.Button("Аргуудыг харьцуулах (Boolean vs Vector Space vs Expanded)", 
                   id='experiment-button',
                   style={'padding': '10px 20px', 'cursor': 'pointer'}),
        dcc.Graph(id='experiment-chart', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'marginBottom': '20px', 'borderRadius': '5px'}),
    
    html.Div([
        html.H3("📈 Word Cloud & TF-IDF График"),
        html.Img(id='wordcloud-img', style={'maxWidth': '800px', 'width': '100%'}),
        dcc.Graph(id='tfidf-chart')
    ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '5px'})
    
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f5f5f5'})

@app.callback(
    Output('init-status', 'children'),
    Input('init-button', 'n_clicks'),
    prevent_initial_call=True
)
def initialize_system(n_clicks):
    global search_engine, initialization_done
    
    if initialization_done:
        return "✅ Систем аль хэдийн ажиллаж байна!"
    
    urls_to_try = [
        'https://en.wikipedia.org/wiki/Information_retrieval',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Natural_language_processing'
    ]
    
    total_docs = 0
    doc_collection.scraping_log = []
    
    for url in urls_to_try:
        num_docs = doc_collection.scrape_website(url)
        total_docs += num_docs
        if total_docs >= 20:  
            break
    
    if total_docs == 0:
        print("⚠️ Вебээс хуулж чадсангүй, жишээ өгөгдлийг уншиж байна...")
        doc_collection.add_sample_documents()
        total_docs = len(doc_collection.documents)
        message_suffix = "(using sample documents - scraping failed)"
    else:
        message_suffix = "(scraped from web)"
    
    # ИНДЕКС ҮҮСГЭХ
    doc_collection.build_inverted_index()
    doc_collection.build_tfidf_matrix()
    
    search_engine = SearchEngine(doc_collection)
    initialization_done = True
    
    log_display = html.Div([
        html.P(f"✅ Системийг ажиллаж эхэлсэн! Ачаалласан {total_docs} баримтууд {message_suffix}"),
        html.Details([
            html.Summary("Scrap-н логийг харах", style={'cursor': 'pointer', 'color': '#3498db'}),
            html.Div([html.P(log, style={'fontSize': '12px', 'margin': '2px'}) 
                     for log in doc_collection.scraping_log])
        ], style={'marginTop': '10px'})
    ])
    
    return log_display

@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('search-method', 'value')],
    prevent_initial_call=True
)
def perform_search(n_clicks, query, method):
    if not initialization_done or not query:
        return "⚠️ Системийг ачаалласны дараа асуулгаа оруулна уу."
    
    if method == 'boolean':
        doc_ids = search_engine.boolean_search(query)
        results = [{'doc_id': doc_id, 
                   'title': doc_collection.documents[doc_id]['title'],
                   'content': doc_collection.documents[doc_id]['content'][:200] + "..."}
                  for doc_id in doc_ids[:10]]
    else:  
        results = search_engine.vector_space_search(query, top_k=10)
    
    if not results:
        return html.Div([
            html.P("Хайлт илэрсэнгүй.", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
            html.P(f"Хайсан: '{query}'"),
            html.P("Өөр түлхүүр үг сонгоно уу.")
        ])
    
    result_divs = [html.H4(f"Олсон {len(results)} үр дүн:")]
    for i, res in enumerate(results):
        score_text = f" (Оноо: {res['score']:.3f})" if 'score' in res else ""
        result_divs.append(
            html.Div([
                html.H4(f"{i+1}. Баримтын ID: {res['doc_id']} - {res['title']}{score_text}"),
                html.P(res['content'], style={'color': '#555'})
            ], style={'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '10px'})
        )
    
    return html.Div(result_divs)

@app.callback(
    Output('expanded-query', 'children'),
    [Input('expand-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('relevant-docs', 'value')],
    prevent_initial_call=True
)
def expand_query(n_clicks, query, relevant_docs_str):
    if not initialization_done or not query or not relevant_docs_str:
        return "⚠️ Асуулга болон холбогдох баримт бичгийн ID-г оруулна уу."
    
    try:
        relevant_docs = [int(x.strip()) for x in relevant_docs_str.split(',')]
        expanded = search_engine.query_expansion_rocchio(query, relevant_docs)
        
        return html.Div([
            html.P(f"Original Query: {query}", style={'fontWeight': 'bold'}),
            html.P(f"Expanded Query: {expanded}", style={'color': '#27ae60', 'fontWeight': 'bold'})
        ])
    except:
        return "⚠️ Баримтын ID хаягийн дугаар буруу байна. Тоог таслалтай оруулна уу. (e.g., 0,2,5)"

@app.callback(
    Output('metrics-display', 'children'),
    [Input('eval-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('search-method', 'value'),
     State('ground-truth', 'value')],
    prevent_initial_call=True
)
def calculate_metrics(n_clicks, query, method, ground_truth_str):
    if not initialization_done or not query or not ground_truth_str:
        return "⚠️ Enter query and ground truth relevant documents."
    
    try:
        ground_truth = [int(x.strip()) for x in ground_truth_str.split(',')]
        
        if method == 'boolean':
            retrieved = search_engine.boolean_search(query)
        else:
            results = search_engine.vector_space_search(query, top_k=10)
            retrieved = [r['doc_id'] for r in results]
        
        k_values = [5, 10]
        metrics = {}
        
        for k in k_values:
            p = evaluator.precision_at_k(retrieved, ground_truth, k)
            r = evaluator.recall_at_k(retrieved, ground_truth, k)
            f1 = evaluator.f1_score(p, r)
            ndcg = evaluator.ndcg_at_k(retrieved, ground_truth, k)
            
            metrics[k] = {'precision': p, 'recall': r, 'f1': f1, 'ndcg': ndcg}
        
        ap = evaluator.average_precision(retrieved, ground_truth)
        
        return html.Div([
            html.H4("Evaluation үр дүн:", style={'color': '#2c3e50'}),
            html.Table([
                html.Tr([html.Th("Үзүүлэлт"), html.Th("@5"), html.Th("@10")],
                       style={'backgroundColor': '#ecf0f1'}),
                html.Tr([html.Td("Precision"), 
                        html.Td(f"{metrics[5]['precision']:.3f}"),
                        html.Td(f"{metrics[10]['precision']:.3f}")]),
                html.Tr([html.Td("Recall"), 
                        html.Td(f"{metrics[5]['recall']:.3f}"),
                        html.Td(f("{metrics[10]['recall']:.3f}"))]),
                html.Tr([html.Td("F1 Score"), 
                        html.Td(f"{metrics[5]['f1']:.3f}"),
                        html.Td(f("{metrics[10]['f1']:.3f}"))]),
                html.Tr([html.Td("NDCG"), 
                        html.Td(f"{metrics[5]['ndcg']:.3f}"),
                        html.Td(f("{metrics[10]['ndcg']:.3f}"))])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px',
                     'border': '1px solid #ddd'}),
            html.P(f"Mean Average Precision (MAP): {ap:.3f}", 
                  style={'marginTop': '15px', 'fontWeight': 'bold', 'color': '#2c3e50'})
        ])
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

@app.callback(
    Output('experiment-chart', 'figure'),
    Input('experiment-button', 'n_clicks'),
    State('query-input', 'value'),
    prevent_initial_call=True
)
def run_experiments(n_clicks, query):
    if not initialization_done or not query:
        return {}
    
    vector_results = search_engine.vector_space_search(query, top_k=10)
    ground_truth = [r['doc_id'] for r in vector_results[:3]]
    
    methods = {}
    
    bool_results = search_engine.boolean_search(query)
    methods['Boolean AND'] = {
        'precision': evaluator.precision_at_k(bool_results, ground_truth, 10),
        'recall': evaluator.recall_at_k(bool_results, ground_truth, 10),
        'ndcg': evaluator.ndcg_at_k(bool_results, ground_truth, 10)
    }
    
    vec_retrieved = [r['doc_id'] for r in vector_results]
    methods['Vector Space'] = {
        'precision': evaluator.precision_at_k(vec_retrieved, ground_truth, 10),
        'recall': evaluator.recall_at_k(vec_retrieved, ground_truth, 10),
        'ndcg': evaluator.ndcg_at_k(vec_retrieved, ground_truth, 10)
    }
    
    if vector_results:
        relevant_docs = [vector_results[0]['doc_id']]  # Use top result
        expanded_query = search_engine.query_expansion_rocchio(query, relevant_docs)
        exp_results = search_engine.vector_space_search(expanded_query, top_k=10)
        exp_retrieved = [r['doc_id'] for r in exp_results]
        methods['Query Expansion'] = {
            'precision': evaluator.precision_at_k(exp_retrieved, ground_truth, 10),
            'recall': evaluator.recall_at_k(exp_retrieved, ground_truth, 10),
            'ndcg': evaluator.ndcg_at_k(exp_retrieved, ground_truth, 10)
        }
    
    method_names = list(methods.keys())
    metrics_list = ['precision', 'recall', 'ndcg']
    
    fig = go.Figure()
    
    for metric in metrics_list:
        values = [methods[m][metric] for m in method_names]
        fig.add_trace(go.Bar(name=metric.upper(), x=method_names, y=values))
    
    fig.update_layout(
        title="Хайлтын аргуудын харьцуулалт",
        xaxis_title="Method",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

@app.callback(
    [Output('wordcloud-img', 'src'),
     Output('tfidf-chart', 'figure')],
    Input('init-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_visualizations(n_clicks):
    if not initialization_done:
        return '', {}
    
    all_text = ' '.join([doc['content'] for doc in doc_collection.documents])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    wordcloud_path = 'assets/wordcloud.png'
    wordcloud.to_file(wordcloud_path)
    
    if doc_collection.tfidf_matrix is not None:
        feature_names = doc_collection.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = doc_collection.tfidf_matrix.sum(axis=0).A1
        top_indices = tfidf_scores.argsort()[-15:][::-1]
        
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [tfidf_scores[i] for i in top_indices]
        
        fig = px.bar(x=top_words, y=top_scores, 
                    labels={'x': 'Terms', 'y': 'TF-IDF Score'},
                    title='Top 15 Terms by TF-IDF Score')
        fig.update_layout(xaxis_tickangle=-45)
        
        return f'/assets/wordcloud.png?t={int(time.time())}', fig
    
    return f'/assets/wordcloud.png?t={int(time.time())}', {}

# RUN 

if __name__ == '__main__':
    print("port ажиллаж байгаа дараад ороорой https://127.0.0.1:1054")
    
    app.run(debug=True, port=1054)
