import vertexai
import requests

from vertexai.preview.language_models import TextGenerationModel

PROJECT_ID = "ibm-keras"
LOCATION = "us-west1" #e.g. us-central1

vertexai.init(project=PROJECT_ID, location=LOCATION)
def predict_large_language_model_sample(
        project_id: str,
        model_name: str,
        temperature: float,
        max_decode_steps: int,
        top_p: float,
        top_k: int,
        content: str,
        location: str = "us-central1",
        tuned_model_name: str = "",
):
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p, )
    print(f"Response from Model: {response.text}")
    return response.text


api_key = ''  # replace with your actual API key
company_list = ['Tesla', 'Apple', 'Google']


def get_financial_news(company_name, api_key):
    url = f"https://api.goperigon.com/v1/all?source=cnn.com&q={company_name}&from=2023-05-18&sortBy=date&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return
    data = response.json()
    articles = data.get('articles')
    if articles is None:
        print("No articles found in the response.")
        return
    for article in articles[:5]:
        print(f"Title: {article.get('title')}")
        print(f"Description: {article.get('description')}")
        print(f"URL: {article.get('url')}")
        print("\n")
    return articles[:5]


from dataclasses import dataclass


@dataclass
class CompanySentiment:
    url: str
    name: str
    sentiment: str


company_sentiment_list = []
for company_name in company_list:
    print("The company name is", company_name)
    articles = get_financial_news(company_name, api_key)
    for article in articles:
        news_article = '''input: ''' + article.get(
            'title') + '''Is the sentiment positive, negative or neutral of the given input? : '''
        print("Constructed news article is", news_article)
        sentiment = predict_large_language_model_sample("ibm-keras", "text-bison@001", 0.2, 5, 0.8, 1, news_article,
                                                        "us-central1")
        print("sentiment of the ", article.get('url'), "is ", sentiment)
        company_sentiment_list.append(CompanySentiment(article.get('url'), company_name, sentiment))
