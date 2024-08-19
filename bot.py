import os
from rasa.nlu.training_data import load_data
from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.core.agent import Agent
from rasa.core.policies import MemoizationPolicy, KerasPolicy
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.core import config as core_config
from rasa.core.train import train
from rasa.core.run import serve_application

# Define NLU training data
nlu_data = """
version: "3.0"

nlu:
- intent: ask_about_kadalundi_disaster
  examples: |
    - What was the Kadalundi train disaster?
    - Tell me about the Kadalundi train disaster
    - What happened in Kadalundi in 2001?
    - Can you explain the Kadalundi train accident?

- intent: ask_about_kadalundi_bridge
  examples: |
    - Can you tell me more about the Kadalundi bridge?
    - Where is the Kadalundi bridge located?
    - Tell me about the Kadalundi bridge
    - What are the coordinates of the Kadalundi bridge?

- intent: ask_about_kadalundi_geography
  examples: |
    - Where is the Kadalundi river?
    - What are the geographical details of the Kadalundi bridge?
    - Which districts does the Kadalundi bridge connect?
"""

# Save NLU data to file
os.makedirs("data", exist_ok=True)
with open("data/nlu.yml", "w") as f:
    f.write(nlu_data)

# Define domain configuration
domain_data = """
version: "3.0"

intents:
  - ask_about_kadalundi_disaster
  - ask_about_kadalundi_bridge
  - ask_about_kadalundi_geography

responses:
  utter_ask_about_kadalundi_disaster:
    - text: |
        The Kadalundi train disaster occurred in India on June 22, 2001, when the Mangalore-Chennai Mail passenger train was crossing over the Kadalundi river.
        During the incident, three carriages fell into the water, resulting in 59 people reported killed or missing, and up to 300 believed injured.
        The official inquiry attributed the derailment to one pillar of the 140-year-old bridge sinking into the riverbed due to recent heavy rain.
        However, subsequent private investigations have challenged this theory, suggesting other possible causes.
        You can read more about it [here](https://en.wikipedia.org/wiki/Kadalundi_train_derailment).

  utter_ask_about_kadalundi_bridge:
    - text: |
        The Kadalundi bridge, built by the British, is the oldest rail link in the region, constructed using screwpile technology.
        The bridge is located at coordinates 11°07′43″N 75°49′53″E, crossing the Kadalundi river near the boundary between Malappuram and Kozhikode districts in Kerala.
        More about the bridge can be found [here](https://yappe.in/kerala/kadalundi/kadalundi-kadavu-bridge/843732).

  utter_ask_about_kadalundi_geography:
    - text: |
        The Kadalundi river forms the borderline between the Malappuram and Kozhikode districts of Kerala.
        The Kadalundi bridge is located at coordinates 11°07′43″N 75°49′53″E.
        Read more details [here](https://frontline.thehindu.com/the-nation/article30251137.ece).
"""

# Save domain data to file
with open("domain.yml", "w") as f:
    f.write(domain_data)

# Define stories configuration
stories_data = """
version: "3.0"

stories:
- story: provide information about Kadalundi disaster
  steps:
  - intent: ask_about_kadalundi_disaster
  - action: utter_ask_about_kadalundi_disaster

- story: provide information about Kadalundi bridge
  steps:
  - intent: ask_about_kadalundi_bridge
  - action: utter_ask_about_kadalundi_bridge

- story: provide information about Kadalundi geography
  steps:
  - intent: ask_about_kadalundi_geography
  - action: utter_ask_about_kadalundi_geography
"""

# Save stories data to file
with open("data/stories.yml", "w") as f:
    f.write(stories_data)

# Define NLU configuration
nlu_config_data = """
language: en

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1
"""

# Save NLU configuration to file
os.makedirs("config", exist_ok=True)
with open("config/nlu_config.yml", "w") as f:
    f.write(nlu_config_data)

# Train the NLU model
trainer = Trainer(config.load("config/nlu_config.yml"))
training_data = load_data("data/nlu.yml")
interpreter = trainer.train(training_data)
model_directory = trainer.persist("./models/nlu", fixed_model_name="nlu_model")

# Define the policies and configuration for core model
core_config_data = """
policies:
  - name: MemoizationPolicy
  - name: KerasPolicy
    epochs: 100
  - name: MappingPolicy
"""

# Save core configuration to file
with open("config/core_config.yml", "w") as f:
    f.write(core_config_data)

# Train the core model
agent = Agent("domain.yml", policies=core_config.load("config/core_config.yml"))
train(
    domain="domain.yml",
    config="config/core_config.yml",
    training_files="data/stories.yml",
    output="models/core",
)

# Run the Rasa chatbot
serve_application(model_directory, interpreter, "domain.yml")