import os

OPENAI_API_KEY = str = os.getenv("OPENAI_API_KEY")
if (OPENAI_API_KEY == None):
    OPENAI_API_KEY = "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"

OPENAI_ORGANIZATION = str = os.getenv("OPENAI_ORGANIZATION")
if (OPENAI_ORGANIZATION == None):
    OPENAI_ORGANIZATION = 'org-HnLCVevNqvvHbAI2GpAtPCw6'

MONGODB_URI = str = os.getenv("MONGODB_URI")
if (MONGODB_URI == None):
    MONGODB_URI = "mongodb://localhost:27017"

DATABASE_NAME = str = os.getenv("DATABASE_NAME")
if (DATABASE_NAME == None):
    DATABASE_NAME = "hbot"

DETAILED_LOGS = os.getenv("DETAILED_LOGS")
DETAILED_LOGS = True if DETAILED_LOGS == "True" else False

COMPANY_ID = None

PORTKEY_API_KEY = str = os.getenv("PORTKEY_API_KEY")
if (PORTKEY_API_KEY == None):
    PORTKEY_API_KEY = 'ezUsxCwPlfpqWg23zaqZsF8ZD8E='

MEMORY_TOKEN_LIMIT = int = os.getenv("MEMORY_TOKEN_LIMIT")
if (MEMORY_TOKEN_LIMIT == None):
    MEMORY_TOKEN_LIMIT = 1000

MONGODB_VECTORSEARCH_URI = str = os.getenv("MONGODB_VECTORSEARCH_URI")
if (MONGODB_VECTORSEARCH_URI == None):
    MONGODB_VECTORSEARCH_URI = "mongodb+srv://QjC9jjAk5stKgPJm:QjC9jjAk5stKgPJm@chatbotgpt.eg05kll.mongodb.net/?retryWrites=true&w=majority&appName=ChatbotGPT"

MONGODB_VECTORSEARCH_DATABASE_NAME = str = os.getenv("MONGODB_VECTORSEARCH_DATABASE_NAME")
if (MONGODB_VECTORSEARCH_DATABASE_NAME == None):
    MONGODB_VECTORSEARCH_DATABASE_NAME = "ChatbotGPT"

MONGODB_VECTORSEARCH_COLLECTION_NAME = str = os.getenv("MONGODB_VECTORSEARCH_COLLECTION_NAME")
if (MONGODB_VECTORSEARCH_COLLECTION_NAME == None):
    MONGODB_VECTORSEARCH_COLLECTION_NAME = "vectorSearch"

ENABLE_REINFORCE_PROMPT = str = os.getenv("ENABLE_REINFORCE_PROMPT")
ENABLE_REINFORCE_PROMPT = True if ENABLE_REINFORCE_PROMPT == "True" else False
if (ENABLE_REINFORCE_PROMPT == None):
    ENABLE_REINFORCE_PROMPT = True

ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION = str = os.getenv("ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION")
ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION = True if ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION == "True" else False
if (ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION == None):
    ENABLE_ATTEMPT_TO_CHANGE_DATA_VERIFICATION = True

TEMPERATURE = os.getenv("TEMPERATURE")
if (TEMPERATURE == None):
    TEMPERATURE = 0.5
else:
    TEMPERATURE = float(TEMPERATURE)

FREQUENCY_PENALTY = os.getenv("FREQUENCY_PENALTY")
if (FREQUENCY_PENALTY == None):
    FREQUENCY_PENALTY = 1
else:
    FREQUENCY_PENALTY = float(FREQUENCY_PENALTY)

GOOGLE_API_KEY = str = os.getenv("GOOGLE_API_KEY")
if (GOOGLE_API_KEY == None):
    # GOOGLE_API_KEY = "AIzaSyDZ7BtcUDKlBeGK2LaksbiYZKHFXHWS_jo"
    GOOGLE_API_KEY = "AIzaSyBsgCgtPDG9Jwo-lmDpS8KXX9obpb0x3Fk"