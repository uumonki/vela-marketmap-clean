from bertopic import BERTopic
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from typing import *


class Atlas:
    def __init__(self, model_path: str, data_path: str, topic_names_path: str):
        # Load model and data
        self.topic_model = BERTopic.load(model_path)
        self.topic_model.verbose = False

        self.companies = pd.read_csv(data_path)
        self.topic_names = pd.read_csv(topic_names_path)

        # Create a dictionary of topic indices and names
        topic_dict = self.topic_model.get_topics()
        for key in topic_dict.keys():
            topic_dict[key] = self.get_topic_name(key)
        self.topic_model.set_topic_labels(topic_dict)

        
    def get_topic_name(self, topic: Union[int, List[int]]) -> Union[int, List[int]]:
        """
        Takes one or more topic indices as input and returns the names.
        """
        if isinstance(topic, int):
            if topic == -1:
                return 'Other'
            return self.topic_names[self.topic_names['topic'] == topic]['name'].values[0]
        elif isinstance(topic, list):
            return [self.get_topic_name(t) for t in topic]


    def classify_index(self, description: Union[str, List[str]]) -> Union[Tuple[int, float], Tuple[List[int], List[float]]]:
        """
        Accepts a string or list of strings as company descriptions.
        Returns a tuple consisting of the topic indices along with confidence scores.
        """
        if isinstance(description, str):
            topic, prob = self.topic_model.transform(description)
            if topic[0] != -1:
                return (topic[0].item(), prob[0][topic[0]].item())
            else:
                if len(topic) > 1:
                    return (topic[1].item(), prob[0][topic[1]].item())
                else:
                    # Return the index with the highest probability
                    return (np.argmax(prob[0]).item(), np.max(prob[0]).item())
        elif isinstance(description, list):
            topics, probs = [], []
            for desc in description:
                topic, prob = self.classify_index(desc)
                topics.append(topic)
                probs.append(prob)
            return (topics, probs)
            

    def classify(self, description: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Returns the topic name(s) given description(s).
        """
        if isinstance(description, str):
            (topic, _) = self.classify_index(description)
            return self.get_topic_name(topic)
        elif isinstance(description, list):
            return [self.classify(desc) for desc in description]
    

    def get_topic_companies(self, topic: int) -> pd.DataFrame:
        """
        Returns a dataframe of all company names given a topic index.
        """
        return self.companies[self.companies['topic'] == topic][['name', 'description']]
    

    def get_company_info(self, company: Union[int, List[int]]) -> pd.DataFrame:
        """
        Returns a dataframe of company information given one or more company indices.
        """
        if isinstance(company, int):
            return self.companies.iloc([company])
        elif isinstance(company, list):
            return self.companies.iloc(company)
    

    def search_topics(self, keyword: str) -> Tuple[List[int], List[str]]:
        """
        Searches for a topic given a keyword. Returns a tuple consisting of the 
        most likely topic indices and names.
        """
        search = self.topic_model.find_topics(keyword)
        return (search[0], self.get_topic_name(search[0]))
    

    def visualize_topics(self, topics: List[int] = None, top_n_topics: int = None) -> go.Figure:
        """
        Visualizes the topic given a topic index.
        """
        return self.topic_model.visualize_topics(custom_labels=True, topics=topics, top_n_topics=top_n_topics)
    

        